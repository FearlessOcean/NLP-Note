```python
def train():

    #1.确定设备
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #2 加载数据集
    dataloader = get_dataloader()

    #3模型
    #获取词表大小
    with open(config.MODELS_DIR / 'vocab.txt','r',encoding='utf-8') as f:

        vocab_list= [line.strip() for line in f.readlines()]

    model = InputMethodModel(vocab_size=len(vocab_list))

  

    #4损失函数
    loss_fn=torch.nn.CrossEntropyLoss()

    #5优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)

  
    #准备 Tensorboard writer   不同层级用时间定义
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H_%M_%S"))

    #定义一个最好的loss作比较
    best_loss=float('inf')

  
    #6训练
    for epoch in range(1,1+config.EPOCH):

        print("="*10,f'Epoch:{epoch}',"="*10)
        #训练一个epoch的逻辑
        loss=tran_one_epoch(
            model,
            dataloader,
            loss_fn,
            optimizer,
            device)
        print(f"loss:{loss}")
        
        #记录损失、记录训练结果
        writer.add_scalar("loss",loss,epoch)


        #保存模型
        if loss < best_loss:
            best_loss=loss
            torch.save(model.state_dict(),config.MODELS_DIR / 'best.pt')
            print("Save Model Successful")
    writer.close()
```
这段 `train()` 就是把你前面所有模块真正串起来的 **训练主流程**。  
可以理解为：**搭好训练现场，然后一轮一轮训练，并保存最优模型。**

我按顺序给你过一遍。

---

## 1. 确定设备

```python
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

意思：

- 如果能用 GPU，就用 `cuda`
    
- 否则就用 `cpu`
    

后面模型和数据都要放到这个 `device` 上。

---

## 2. 加载数据集

```python
dataloader = get_dataloader()
```

这里假设你已经封装好了 `get_dataloader()`，它会返回一个 `DataLoader`。

作用：

- 按 batch 取数据
    
- 训练时一批一批喂给模型
    

---

## 3. 创建模型

### 先读词表

```python
with open(config.MODELS_DIR / 'vocab.txt','r',encoding='utf-8') as f:
    vocab_list= [line.strip() for line in f.readlines()]
```

这里是在读取词表文件。

比如 `vocab.txt` 里每行一个词：

```python
我
爱
你
中
国
...
```

读完后：

```python
vocab_list
```

就是一个列表。

### 再确定词表大小

```python
len(vocab_list)
```

这就是 `vocab_size`。

### 创建模型

```python
model = InputMethodModel(vocab_size=len(vocab_list))
```

意思：

- Embedding 层要知道词表有多大
- 最后 Linear 输出也要知道有多少个类别

---

## 4. 定义损失函数

```python
loss_fn=torch.nn.CrossEntropyLoss()
```

这是多分类里最常用的损失函数。

对应你的模型输出：

- `outputs.shape = [batch_size, vocab_size]`
    

对应你的标签：

- `targets.shape = [batch_size]`
    

也就是：

- 模型输出每个词的分数
    
- target 给出正确词的编号
    

---

## 5. 定义优化器

```python
optimizer = torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)
```

意思：

- 让 Adam 去更新模型参数
    
- `model.parameters()` 包括：
    
    - embedding 的参数
        
    - rnn 的参数
        
    - linear 的参数
        

---

## 6. TensorBoard 记录器

```python
writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H_%M_%S"))
```

作用：

把训练过程中的指标写到日志里，后面可以用 TensorBoard 看曲线。

这里日志目录用时间命名，是为了：

- 每次训练单独一个文件夹
    
- 不和之前记录混在一起
    

---

## 7. 初始化最优 loss

```python
best_loss=float('inf')
```

意思：

一开始把“最佳 loss”设成正无穷。

这样第一轮训练出来的 loss 一定比它小，就能顺利保存第一版模型。

---

## 8. 开始训练循环

```python
for epoch in range(1,1+config.EPOCH):
```

如果 `config.EPOCH = 10`，那就训练 10 轮。

这里从 1 开始写，是为了打印 epoch 更直观。

---

## 9. 训练一个 epoch

```python
loss=tran_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device)
```

这一句会做完一整轮训练。

也就是：

- dataloader 里的所有 batch 都跑一遍
    
- 每个 batch 前向传播、算 loss、反向传播、更新参数
    

返回值 `loss` 是这一轮的平均 loss。

---

## 10. 打印 loss

```python
print(f"loss:{loss}")
```

就是输出当前这一轮的训练损失。

---

## 11. 写入 TensorBoard

```python
writer.add_scalar("loss",loss,epoch)
```

作用：

把当前 epoch 的 loss 记录下来。

后面你在 TensorBoard 里就能看到 loss 曲线。

---

## 12. 保存最优模型

```python
if loss < best_loss:
    best_loss=loss
    torch.save(model.state_dict(),config.MODELS_DIR / 'best.pt')
    print("Save Model Successful")
```

意思是：

- 如果这一轮 loss 比历史最好还小
    
- 就更新 `best_loss`
    
- 并把当前模型参数保存到 `best.pt`
    

这里保存的是：

```python
model.state_dict()
```

也就是模型参数，不是整个模型对象。

这是 PyTorch 很常见、也更推荐的保存方式。

---

## 13. 关闭 writer

```python
writer.close()
```

训练结束后关闭 TensorBoard 写入器。

这是个好习惯。

---

# 整个流程一句话串起来

这段 `train()` 做的事情是：

**准备设备、数据、模型、损失函数、优化器和日志工具，然后循环多个 epoch 训练模型，记录 loss，并保存 loss 最小的模型参数。**

---

# 这段代码非常标准的训练框架可以概括成

```python
准备 device
准备 dataloader
准备 model
准备 loss_fn
准备 optimizer
准备日志工具
for epoch in epochs:
    训练一轮
    记录指标
    保存最好模型
```

---

# 你这段代码里有两个小提醒

## 1. `model.to(device)` 更常放在这里

你现在是在 `tran_one_epoch()` 里做的：

```python
model = model.to(device)
```

更常见写法是，在 `train()` 里模型创建后就放上去：

```python
model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
```

这样更清晰，也不用每个 epoch 重复放。

---

## 2. 目前只有训练集，没有验证集

你这里的“best model”其实是按**训练 loss** 保存的。

严格来说，更常见做法是：

- 训练集负责更新参数
    
- 验证集负责选最优模型
    

但你现在作为入门训练流程，这么写是可以的。

---

# 一句话总结

**这段 `train()` 是完整训练入口：初始化训练所需组件，循环训练多个 epoch，记录 loss，并保存当前训练 loss 最低的模型。**

你下一句最适合问我：  
**`state_dict()` 到底是什么，为什么保存它而不是直接保存 model？**