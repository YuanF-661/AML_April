def train_with_distillation(model, train_loader, val_loader, teacher_model,
                            optimizer, device, num_epochs,
                            alpha=0.5, temperature=2.0, use_raw_audio=True):
    """
    使用知识蒸馏训练模型

    Args:
        model: 学生模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        teacher_model: PANNs教师模型
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数
        alpha: 硬目标和软目标损失的权重平衡参数
        temperature: 软化教师预测的温度参数
        use_raw_audio: 是否使用原始音频波形进行蒸馏

    Returns:
        训练历史记录
    """



    print(f"开始训练，使用知识蒸馏 (alpha={alpha}, temperature={temperature})")
    model.to(device)

    # 在创建学生模型之后添加此检查
    def check_requires_grad(model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"警告: 参数 {name} 不需要梯度！")
                param.requires_grad = True

    # 使用
    check_requires_grad(model)

    # 普通分类损失
    criterion = nn.CrossEntropyLoss()

    # 记录训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_type_acc': [], 'val_type_acc': [],
        'train_machine_acc': [], 'val_machine_acc': [],
        'distill_loss': []
    }

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        distill_loss_sum = 0.0
        type_correct = 0
        machine_correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            # 批量数据处理
            if len(batch) == 2 and not use_raw_audio:
                # 标准格式：(mfcc, (type_labels, machine_labels))
                mfcc, (type_labels, machine_labels) = batch

                # 将MFCC转换为波形（近似）
                waveform = convert_mfcc_to_waveform(mfcc, sr=sample_rate)

            elif len(batch) == 2 and use_raw_audio:
                # 自定义格式：(mfcc, waveform, (type_labels, machine_labels))
                # 注意：这需要修改数据集的__getitem__方法
                raise NotImplementedError("需要修改数据集类以支持原始波形")

            elif len(batch) == 3:
                # one-hot增强格式：(mfcc, (type_labels, machine_labels, type_onehot))
                mfcc, (type_labels, machine_labels, type_onehot) = batch

                # 将MFCC转换为波形（近似）
                waveform = convert_mfcc_to_waveform(mfcc, sr=sample_rate)

            else:
                raise ValueError(f"不支持的批量格式: {len(batch)}")

            # 数据移到设备
            mfcc = mfcc.to(device)
            waveform = waveform.to(device)
            type_labels = type_labels.to(device)
            machine_labels = machine_labels.to(device)

            # 先获取鼓类型和鼓机型号的数量
            num_drum_types = len(torch.unique(type_labels))
            num_drum_machines = len(torch.unique(machine_labels))

            # 获取教师模型的预测
            with torch.no_grad():
                teacher_outputs = teacher_model.predict(waveform)

                # 将教师输出转换为可以在计算图中使用但不计算梯度的张量
                teacher_type_logits = teacher_outputs[:, :num_drum_types].clone().detach()
                teacher_machine_logits = teacher_outputs[:, -num_drum_machines:].clone().detach()

                # 确保维度匹配
                if teacher_type_logits.size(1) < num_drum_types:
                    padding = torch.zeros(teacher_type_logits.size(0),
                                          num_drum_types - teacher_type_logits.size(1),
                                          device=device)
                    teacher_type_logits = torch.cat([teacher_type_logits, padding], dim=1)

                if teacher_machine_logits.size(1) < num_drum_machines:
                    padding = torch.zeros(teacher_machine_logits.size(0),
                                          num_drum_machines - teacher_machine_logits.size(1),
                                          device=device)
                    teacher_machine_logits = torch.cat([teacher_machine_logits, padding], dim=1)

            # 前向传播（学生模型）
            if 'type_onehot' in locals():
                type_onehot = type_onehot.to(device)
                student_type_logits, student_machine_logits = model(mfcc, type_onehot)
            else:
                student_type_logits, student_machine_logits = model(mfcc)

            # 计算硬目标损失（常规交叉熵）
            type_ce_loss = criterion(student_type_logits, type_labels)
            machine_ce_loss = criterion(student_machine_logits, machine_labels)
            hard_loss = type_ce_loss + machine_ce_loss

            # 在计算知识蒸馏损失前添加
            # print(
            #     f"批次 {batch_idx}: Student type: {student_type_logits.shape}, Teacher type: {teacher_type_logits.shape}")
            # print(
            #     f"批次 {batch_idx}: Student machine: {student_machine_logits.shape}, Teacher machine: {teacher_machine_logits.shape}")

            # 计算知识蒸馏损失
            type_distill_loss = distillation_loss(
                student_type_logits, teacher_type_logits.detach(),  # 明确detach教师输出
                labels=None, alpha=0, temperature=temperature
            )

            machine_distill_loss = distillation_loss(
                student_machine_logits, teacher_machine_logits.detach(),  # 明确detach教师输出
                labels=None, alpha=0, temperature=temperature
            )

            soft_loss = type_distill_loss + machine_distill_loss

            # 组合硬目标和软目标损失
            loss = alpha * hard_loss + (1 - alpha) * soft_loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            distill_loss_sum += soft_loss.item()

            _, type_preds = torch.max(student_type_logits, 1)
            _, machine_preds = torch.max(student_machine_logits, 1)

            type_correct += (type_preds == type_labels).sum().item()
            machine_correct += (machine_preds == machine_labels).sum().item()
            total += type_labels.size(0)

            # 显示进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                      f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Distill Loss: {soft_loss.item():.4f}")

        print(f"Student type logits shape: {student_type_logits.shape}")
        print(f"Teacher type logits shape: {teacher_type_logits.shape}")

        # 计算平均损失和准确率
        train_loss /= len(train_loader)
        distill_loss_avg = distill_loss_sum / len(train_loader)
        train_type_acc = 100.0 * type_correct / total
        train_machine_acc = 100.0 * machine_correct / total

        # 保存到历史记录
        history['train_loss'].append(train_loss)
        history['distill_loss'].append(distill_loss_avg)
        history['train_type_acc'].append(train_type_acc)
        history['train_machine_acc'].append(train_machine_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_type_correct = 0
        val_machine_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                # 批量数据处理（与训练阶段相同）
                if len(batch) == 2 and not use_raw_audio:
                    mfcc, (type_labels, machine_labels) = batch
                    waveform = convert_mfcc_to_waveform(mfcc, sr=sample_rate)
                elif len(batch) == 2 and use_raw_audio:
                    raise NotImplementedError("需要修改数据集类以支持原始波形")
                elif len(batch) == 3:
                    mfcc, (type_labels, machine_labels, type_onehot) = batch
                    waveform = convert_mfcc_to_waveform(mfcc, sr=sample_rate)
                else:
                    raise ValueError(f"不支持的批量格式: {len(batch)}")

                # 数据移到设备
                mfcc = mfcc.to(device)
                type_labels = type_labels.to(device)
                machine_labels = machine_labels.to(device)

                # 学生模型前向传播
                if 'type_onehot' in locals():
                    type_onehot = type_onehot.to(device)
                    student_type_logits, student_machine_logits = model(mfcc, type_onehot)
                else:
                    student_type_logits, student_machine_logits = model(mfcc)

                # 计算验证损失（仅使用硬目标损失）
                type_loss = criterion(student_type_logits, type_labels)
                machine_loss = criterion(student_machine_logits, machine_labels)
                loss = type_loss + machine_loss

                val_loss += loss.item()

                # 计算准确率
                _, type_preds = torch.max(student_type_logits, 1)
                _, machine_preds = torch.max(student_machine_logits, 1)

                val_type_correct += (type_preds == type_labels).sum().item()
                val_machine_correct += (machine_preds == machine_labels).sum().item()
                val_total += type_labels.size(0)

        # 计算平均验证损失和准确率
        val_loss /= len(val_loader)
        val_type_acc = 100.0 * val_type_correct / val_total
        val_machine_acc = 100.0 * val_machine_correct / val_total

        # 保存到历史记录
        history['val_loss'].append(val_loss)
        history['val_type_acc'].append(val_type_acc)
        history['val_machine_acc'].append(val_machine_acc)

        # 打印训练信息
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Distill Loss: {distill_loss_avg:.4f}, "
              f"Train Type Acc: {train_type_acc:.2f}%, "
              f"Train Machine Acc: {train_machine_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Type Acc: {val_type_acc:.2f}%, "
              f"Val Machine Acc: {val_machine_acc:.2f}%")

    return history
