import base64
from openai import OpenAI
import os
import json
import re
import io
from flask import Flask, request, jsonify, render_template, session
from PIL import Image
from typing import Optional

# --- 模型推理与辅助函数 ---

# OpenAI 客户端设置 (请确保你的 vLLM 服务器正在运行)
try:
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8007/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

def extract_boxed_content(text: Optional[str]) -> Optional[str]:
    """从文本中提取 \\boxed{} 内的内容"""
    if text is None:
        return None
    match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def run_inference(prompt: str, image: Optional[Image.Image] = None) -> str:
    """
    接收文本和可选的图片，调用模型进行推理。
    返回模型的原始输出字符串。
    """
    if client is None:
        return "错误：OpenAI 客户端未初始化。请检查 vLLM 服务器连接。"

    try:
        # 构建消息体
        user_content = [
            {"type": "text", "text": prompt},
        ]

        # 如果有图片，将其编码并添加到消息中
        if image:
            buffered = io.BytesIO()
            # 确保图片是 RGB 格式
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
            })

        messages = [
            {"role": "system", "content": "You are a helpful assistant."}, # 你可以根据需要修改系统提示
            {
                "role": "user",
                "content": user_content,
            },
        ]

        # 调用模型
        chat_response = client.chat.completions.create(
            model="Qwen2-VL-7B-Instruct", # 确保模型名称正确
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=7000
        )
        response_text_raw = chat_response.choices[0].message.content
        
        # 在后台打印模型的完整输出
        print("--- Model Full Output ---")
        print(response_text_raw)
        print("------------------------")
        
        return response_text_raw

    except Exception as e:
        error_message = f"模型推理时发生错误: {e}"
        print(error_message)
        return error_message

# --- Flask Web 应用 ---

app = Flask(__name__)
# 用于 session 管理，保证每个用户的对话历史是独立的
app.secret_key = os.urandom(24) 

# 创建用于存放上传文件的目录
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def index():
    """渲染主页面"""
    session.clear() # 开始新会话时清空历史记录
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    # 初始化对话历史
    if 'history' not in session:
        session['history'] = []

    user_prompt = request.form.get('prompt')
    image_file = request.files.get('image')

    pil_image = None
    if image_file:
        try:
            pil_image = Image.open(image_file.stream)
        except Exception as e:
            return jsonify({'error': f'无法处理上传的图片: {e}'}), 400

    # 准备给模型的提示 (这里可以像你的脚本一样加入模板)
    # 为了通用性，我们直接使用用户输入，但你可以轻易修改
    # 例如： full_prompt = f"You are an expert in math... Question: {user_prompt}"
    full_prompt = user_prompt

    # 调用推理函数
    model_response = run_inference(full_prompt, pil_image)

    # 更新对话历史
    # 注意：我们不存储图片本身在 session 中，只在当次请求中使用
    session['history'].append({"role": "user", "content": user_prompt})
    session['history'].append({"role": "assistant", "content": model_response})
    
    # 强制 Flask 保存 session
    session.modified = True
    
    # 在后台打印当前完整的对话历史
    print("--- Current Conversation History ---")
    print(json.dumps(session['history'], indent=2, ensure_ascii=False))
    print("----------------------------------")

    return jsonify({'response': model_response})

if __name__ == '__main__':
    # 运行 Flask 应用
    # host='0.0.0.0' 让应用可以被局域网内的其他设备访问
    app.run(host='0.0.0.0', port=5000, debug=True)
