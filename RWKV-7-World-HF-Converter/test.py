import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径
model_path = "/mnt/data/Atomgit/Mobius"

print("=== 正在加载模型和分词器 (CPU模式) ===")

# 1. 加载模型
# device_map="cpu": 强制模型加载到 CPU
# torch_dtype=torch.float32: CPU 上通常使用 float32 兼容性最好
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    device_map="cpu", 
    torch_dtype=torch.float32
)

# 2. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# === 关键修复：手动设置 RWKV World 的 Chat Template ===
# RWKV World 模型通常遵循 "User: xxx\n\nAssistant: xxx\n\n" 的格式。
# 下面的 Jinja2 模板定义了如何将 messages 列表转换为这种字符串。
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ 'User: ' + message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'Assistant: ' + message['content'] + '\n\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'Assistant:' }}"
    "{% endif %}"
)

# === 检查权重信息 ===
print("\n=== HuggingFace模型的权重键 ===")
state_dict = model.state_dict()
keys_list = sorted(list(state_dict.keys()))
print(f"模型中共有 {len(keys_list)} 个权重键")
# print(keys_list) # 如果键太多，建议注释掉这行，避免刷屏

print("\n=== 部分权重的形状信息 (前5个) ===")
# 这里我修改为 [:5]，以匹配你注释里的"只打印前5个"，否则会打印所有层
for key in list(state_dict.keys())[:5]: 
    print(f"权重 '{key}' 的形状: {state_dict[key].shape}")

# === 准备推理数据 ===
prompt = "帮我写一个沁园春雪"
messages = [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am a GPT-3 based model."}, # 注意：RWKV其实不是GPT-3 based，这里只是为了复现你的prompt
    {"role": "user", "content": prompt}
]

# 3. 应用聊天模板
# 之前报错是因为 tokenizer 缺失模板，我们在上面手动设置了 tokenizer.chat_template，现在可以正常运行了
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(f"\n=== 构建的 Prompt ===\n{text}")
print("=====================\n")

# 4. 编码输入并确保在 CPU 上
# return_tensors="pt" 返回 PyTorch 张量
# .to("cpu") 确保输入数据也在 CPU 上
model_inputs = tokenizer([text], return_tensors="pt").to("cpu")

print("=== 开始生成 (CPU速度较慢，请耐心等待) ===")

# 5. 生成
# 移除了 **model_inputs 的直接解包，为了更清晰地控制
with torch.no_grad(): # 推理时不需要计算梯度，节省内存
    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=200,
        do_sample=True, # 建议开启采样，生成的回复更自然
        temperature=1.0,
        top_p=0.3
    )

# 6. 处理输出
# 裁剪掉输入的 token，只保留新生成的 token
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"\nresponse is: {response}")