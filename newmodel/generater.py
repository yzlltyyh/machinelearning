import json
import random
import asyncio
import aiohttp
from datetime import datetime
import os

class EmotionDataGenerator:
    def __init__(self, api_key, api_base="https://api.aigclink.xyz", dataset_path="dataset", 
                 min_temperature=0.7, max_temperature=0.9):
        self.api_key = api_key
        self.api_base = api_base
        self.dataset_path = dataset_path
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        os.makedirs(dataset_path, exist_ok=True)

    async def async_call_api(self, messages, session):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gemini-2.0-flash-exp",
            "messages": messages,
            "temperature": random.uniform(self.min_temperature, self.max_temperature),
            "max_tokens": 8096
        }
        
        max_retries = 3
        retry_delay = 1
        
        for retry in range(max_retries):
            try:
                async with session.post(
                    f"{self.api_base}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                ) as response:
                    if response.status == 429:
                        print(f"API频率限制，等待{retry_delay}秒后重试...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                        
                    if response.status != 200:
                        print(f"API响应错误: {await response.text()}")
                        await asyncio.sleep(retry_delay)
                        continue
                        
                    return await response.json()
                    
            except Exception as e:
                print(f"API调用出错: {str(e)}")
                await asyncio.sleep(retry_delay)
        
        return None

    async def generate_batch(self, scenarios, system_prompt, batch_size, session):
        tasks = []
        for _ in range(batch_size):
            scenario_type, scenario_details = random.choice(scenarios)
            specific_scenario = random.choice(scenario_details)
            
            user_prompt = f"""
            请生成一个关于{scenario_type}中{specific_scenario}的场景文本，要求：
            1. 包含隐喻或深层含义
            2. 表达要有文学性和深度
            3. 情感分析要准确反映文本的深层含义
            4. 返回合法的JSON格式
            """
            
            task = self.async_call_api([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], session)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

    async def async_generate_data(self, num_samples=500):
        texts = []
        labels = []
        
        scenarios = [
            ("工作情境", [
                "项目压力", "团队协作", "工作成就", "职业发展",
                "工作倦怠", "职场人际", "工作变动", "晋升焦虑",
                "办公室政治", "工作生活平衡"
            ]),
            ("生活情境", [
                "日常琐事", "休闲娱乐", "生活压力", "健康状况",
                "财务状况", "生活规划", "个人习惯", "家庭关系",
                "居住环境", "未来规划"
            ]),
            ("社交情境", [
                "人际互动", "社交活动", "人际冲突", "社交支持",
                "社交焦虑", "社交关系", "群体活动", "社交孤立",
                "友情维系", "社交恐惧"
            ]),
            ("情感情境", [
                "恋爱关系", "家庭关系", "友情维系", "情感困扰",
                "情感期待", "情感失落", "情感交流", "分手创伤",
                "感情迷茫", "自我怀疑"
            ]),
            ("深度表达", [
                "孤独感悟", "生命思考", "存在意义", "人生价值",
                "自我认知", "心灵对话", "情感共鸣", "心理成长",
                "精神追求", "内心探索"
            ]),
            ("隐喻表达", [
                "孤独隐喻", "情感比喻", "反讽表达", "双关语",
                "诗意描述", "深度感悟", "自我对话", "心理投射",
                "情感象征", "意识流"
            ])
        ]
        
        system_prompt = """
        你是一个专业的情感分析专家，精通文学表达和心理分析。请根据给定场景生成一段富有深度的情感文本，包含隐喻、比喻或象征性的表达。请按照以下格式生成响应：
        {
            "text": "这里是情感文本内容（要求有文学性和深度）",
            "scores": {
                "valence": -0.7,  # 情感效价，范围-1到1
                "arousal": 75,    # 情感唤醒度，范围0-100
                "dominance": 80,  # 情感主导性，范围0-100
                "anxiety": 12,    # 焦虑指数，范围0-21
                "depression": 14, # 抑郁指数，范围0-21
                "stress": 25,     # 压力指数，范围0-40
                "emotional_salience": 90,      # 情感显著性，范围0-100
                "emotional_complexity": 65,    # 情感复杂度，范围0-100
                "loneliness_index": 95,       # 孤独指数，范围0-100
                "self_reflection": 85         # 自我反思，范围0-100
            }
        }

        注意：
        1. 文本必须包含隐喻、比喻或深层含义，体现文学性和深度
        2. 所有分数必须严格遵守指定范围
        3. 分数要与文本内容高度匹配，反映文本的情感特征
        4. 响应必须是合法的JSON格式
        """
        
        print(f"开始生成{num_samples}条训练数据...")
        
        batch_size = 20
        successful_samples = 0
        
        async with aiohttp.ClientSession() as session:
            while successful_samples < num_samples:
                current_batch_size = min(batch_size, num_samples - successful_samples)
                responses = await self.generate_batch(scenarios, system_prompt, current_batch_size, session)
                
                for response in responses:
                    if response and self._process_response(response, texts, labels):
                        successful_samples += 1
                        if successful_samples % 10 == 0:
                            print(f"已成功生成 {successful_samples} 条数据")
                
                await asyncio.sleep(1)
        
        print(f"数据生成完成，成功生成 {successful_samples} 条数据")
        return texts, labels

    def _process_response(self, response, texts, labels):
        try:
            generated_content = response['choices'][0]['message']['content'].strip()
            if "```json" in generated_content:
                generated_content = generated_content.split("```json")[1].split("```")[0]
            elif "```" in generated_content:
                generated_content = generated_content.split("```")[1]
            
            response_data = json.loads(generated_content)
            
            required_fields = ['text', 'scores']
            required_scores = ['valence', 'arousal', 'dominance', 'anxiety', 
                             'depression', 'stress', 'emotional_salience', 
                             'emotional_complexity', 'loneliness_index', 'self_reflection']
            
            if all(field in response_data for field in required_fields) and \
               all(score in response_data['scores'] for score in required_scores):
                
                texts.append(response_data['text'])
                labels.append(self._normalize_scores(response_data['scores']))
                return True
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"数据处理错误: {str(e)}")
        return False

    def _normalize_scores(self, scores):
        return [
            scores['valence'],
            scores['arousal'] / 100,
            scores['dominance'] / 100,
            scores['anxiety'] / 21,
            scores['depression'] / 21,
            scores['stress'] / 40,
            scores['emotional_salience'] / 100,
            scores['emotional_complexity'] / 100,
            scores['loneliness_index'] / 100,
            scores['self_reflection'] / 100
        ]

    def generate_data(self, num_samples=50):
        return asyncio.run(self.async_generate_data(num_samples))

    def save_data(self, texts, labels, filename=None):
        if filename is None:
            filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        file_path = os.path.join(self.dataset_path, filename)
        
        dataset = []
        for text, label in zip(texts, labels):
            data_item = {
                "text": text,
                "label": label,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            dataset.append(data_item)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
        print(f"数据已保存至: {file_path}")
        print(f"当前数据集大小: {len(dataset)}条")

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    # 从环境变量获取API密钥
    api_key = os.getenv("EMOTION_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 EMOTION_API_KEY")
    
    # 测试数据生成
    generator = EmotionDataGenerator(api_key)
    texts, labels = generator.generate_data(num_samples=500)  # 测试生成10条数据
    
    # 保存生成的数据
    generator.save_data(texts, labels)