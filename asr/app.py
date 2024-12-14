import os
import uuid
import json
import hashlib
import subprocess
import re
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic_settings import BaseSettings
import google.generativeai as genai
import mimetypes
import tempfile
import openai
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Settings(BaseSettings):
    gemini_api_key: str
    openai_api_key: str
    openai_base_url: str
    temp_dir: str = "temp"
    cache_dir: str = "cache"
    
    class Config:
        env_file = ".env"

class TextSplitter:
    SYSTEM_PROMPT = """
    你是一名句子拆分专家，擅长将没有断句的一整段文本，拆分成一句句文本，每句文本之间用<br>隔开。
    要求：
    1. 不按照完整的句子拆分段落，只需按照语义进行拆分段落，注意保持句子结构的完整性。
    2. 不要修改原句的任何内容，也不要添加任何内容，你只需要每句文本之间添加<br>隔开。
    3. 直接返回拆分后的文本，不要返回任何其他说明内容，不需要任何标点符号。
    4. 对于中文每个拆分句文本总字数不超过40；对于英文每个断句单词(word)总数目不超过40。
    5. 对于asr产生的多余的语气词，或者重复的无意义的字，请忽略，明显的错别字请修正。
    6. 如果是中文的话，输出简体中文。
    """
    
    def __init__(self, api_key: str, base_url: str, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def get_cache_key(self, text: str) -> str:
        return hashlib.md5(f"{text}_gemini-2.0-flash-exp".encode()).hexdigest()
    
    def get_cache(self, text: str) -> Optional[List[str]]:
        cache_key = self.get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding='utf-8'))
            except (IOError, json.JSONDecodeError):
                return None
        return None
    
    def set_cache(self, text: str, result: List[str]) -> None:
        cache_key = self.get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            cache_file.write_text(
                json.dumps(result, ensure_ascii=False),
                encoding='utf-8'
            )
        except IOError:
            pass
    
    async def split_text(self, text: str, use_cache: bool = True) -> List[str]:
        if use_cache:
            cached_result = self.get_cache(text)
            if cached_result:
                return cached_result
        
        prompt = f"请你对下面句子使用<br>进行分割：\n{text}"
        try:
            response = self.client.chat.completions.create(
                model="gemini-2.0-flash-exp",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            result = response.choices[0].message.content
            # 清理结果中的多余换行符
            result = re.sub(r'\n+', '', result)
            split_result = [segment.strip() for segment in result.split("<br>") if segment.strip()]
            
            self.set_cache(text, split_result)
            return split_result
            
        except Exception as e:
            print(f"[!] 文本断句失败: {e}")
            return []

class MediaProcessor:
    SUPPORTED_VIDEO_FORMATS = {
        'video/mp4', 'video/mpeg', 'video/quicktime', 
        'video/x-msvideo', 'video/x-matroska'
    }
    
    SUPPORTED_AUDIO_FORMATS = {
        'audio/wav', 'audio/mp3', 'audio/aiff', 
        'audio/aac', 'audio/ogg', 'audio/flac'
    }
    
    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
    def _check_ffmpeg(self):
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True)
            return True
        except FileNotFoundError:
            return False
    
    def cleanup_temp_files(self, *files):
        for file in files:
            try:
                if file and os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                print(f"Error cleaning up {file}: {e}")
    
    async def process_media(self, file: UploadFile) -> tuple[bytes, str]:
        """处理上传的媒体文件，返回音频内容和MIME类型"""
        if not file.content_type:
            mime_type, _ = mimetypes.guess_type(file.filename)
            if not mime_type:
                raise ValueError("无法确定文件类型")
        else:
            mime_type = file.content_type
            
        # 为临时文件生成唯一文件名
        temp_input_path = self.temp_dir / f"{uuid.uuid4()}{Path(file.filename).suffix}"
        temp_output_path = self.temp_dir / f"{uuid.uuid4()}.mp3"
        
        try:
            # 保存上传的文件
            content = await file.read()
            with open(temp_input_path, "wb") as f:
                f.write(content)
            
            # 如果是视频文件，转换为音频
            if mime_type in self.SUPPORTED_VIDEO_FORMATS:
                if not self._check_ffmpeg():
                    raise ValueError("未安装ffmpeg，无法处理视频文件")
                    
                # 使用ffmpeg提取音频
                command = [
                    'ffmpeg', '-i', str(temp_input_path),
                    '-vn',  # 禁用视频
                    '-acodec', 'libmp3lame',  # 使用MP3编码器
                    '-ar', '44100',  # 采样率
                    '-ac', '2',  # 声道数
                    '-b:a', '192k',  # 比特率
                    str(temp_output_path)
                ]
                
                process = subprocess.run(command, capture_output=True)
                if process.returncode != 0:
                    raise ValueError(f"视频处理失败: {process.stderr.decode()}")
                
                # 读取处理后的音频文件
                with open(temp_output_path, 'rb') as f:
                    audio_content = f.read()
                return audio_content, 'audio/mp3'
                
            # 如果是音频文件，直接返回
            elif mime_type in self.SUPPORTED_AUDIO_FORMATS:
                return content, mime_type
            else:
                raise ValueError(f"不支持的文件格式: {mime_type}")
                
        finally:
            # 清理临时文件
            self.cleanup_temp_files(temp_input_path, temp_output_path)

class GeminiASR:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    async def transcribe_audio(self, file_content: bytes, mime_type: str) -> str:
        try:
            prompt = "生成音频的逐字转写。只输出转写的文本,不要添加任何额外的解释或格式，如果是中文，请转录成简体中文。"
            
            audio_data = {
                "mime_type": mime_type,
                "data": file_content
            }
            
            response = self.model.generate_content([prompt, audio_data])
            return response.text.strip()
            
        except Exception as e:
            raise Exception(f"转写错误: {str(e)}")

app = FastAPI(
    title="音视频转文字 API",
    description="基于 Gemini 的音视频转写服务",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

try:
    settings = Settings()
except Exception:
    raise Exception("请确保已经设置了所有必要的环境变量")

media_processor = MediaProcessor(settings.temp_dir)
asr = GeminiASR(settings.gemini_api_key)
text_splitter = TextSplitter(
    settings.openai_api_key,
    settings.openai_base_url,
    settings.cache_dir
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>音视频转文字 API 文档</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            .endpoint {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }
            .method {
                background: #4CAF50;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 14px;
            }
            code {
                background: #e9ecef;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: monospace;
            }
            .note {
                border-left: 4px solid #ffc107;
                padding-left: 15px;
                margin: 15px 0;
                color: #666;
            }
        </style>
    </head>
    <body>
        <h1>🎙️ 音视频转文字 API 文档</h1>
        
        <p>这是一个基于 Gemini 的音视频转写服务，支持多种格式的音视频转写功能。</p>
        
        <div class="endpoint">
            <span class="method">POST</span>
            <code>/api/v1/transcribe</code>
            
            <h3>接口说明</h3>
            <p>将音视频文件转换为文字。</p>
            
            <h3>请求格式</h3>
            <ul>
                <li>请求方式：POST</li>
                <li>Content-Type: multipart/form-data</li>
                <li>参数名：file（音视频文件）</li>
            </ul>
            
            <h3>支持的视频格式</h3>
            <ul>
                <li>MP4 (.mp4)</li>
                <li>MPEG (.mpeg, .mpg)</li>
                <li>QuickTime (.mov)</li>
                <li>AVI (.avi)</li>
                <li>Matroska (.mkv)</li>
            </ul>
            
            <h3>支持的音频格式</h3>
            <ul>
                <li>WAV (.wav)</li>
                <li>MP3 (.mp3)</li>
                <li>AAC (.aac)</li>
                <li>OGG (.ogg)</li>
                <li>FLAC (.flac)</li>
                <li>AIFF (.aiff)</li>
            </ul>
            
            <h3>限制条件</h3>
            <div class="note">
                <p>• 文件大小上限：1GB</p>
                <p>• 需要安装ffmpeg才能处理视频文件</p>
                <p>• 仅支持上述列出的音视频格式</p>
            </div>
            
            <h3>返回格式</h3>
            <pre><code>{
    "transcript": "转写的文本内容",
    "segments": ["断句后的", "文本段落"]
}</code></pre>
            
            <h3>错误码</h3>
            <ul>
                <li>400：请求参数错误（文件格式不支持、文件过大等）</li>
                <li>500：服务器内部错误</li>
            </ul>
        </div>
        
        <footer style="text-align: center; margin-top: 40px; color: #666;">
            <p>© 2024 音视频转文字 API 服务</p>
        </footer>
    </body>
    </html>
    """

@app.post("/api/v1/transcribe")
async def transcribe_media(file: UploadFile):
    """
    转写音视频文件为文本，并进行断句处理
    """
    try:
        # 检查文件大小
        file_size = 0
        content = await file.read()
        file_size = len(content)
        await file.seek(0)
        
        if file_size > 1024 * 1024 * 1024:  # 1GB
            raise HTTPException(status_code=400, detail="文件大小超过1GB限制")
        
        # 处理音视频
        try:
            audio_content, mime_type = await media_processor.process_media(file)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # 转写文本
        transcript = await asr.transcribe_audio(audio_content, mime_type)
        
        # 对转写文本进行断句
        split_result = await text_splitter.split_text(transcript)
        
        return {
            "transcript": transcript,  # 原始转写文本
            "segments": split_result   # 断句后的文本段落
        }
        
    except Exception as e:
        if "未安装ffmpeg" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=2323)