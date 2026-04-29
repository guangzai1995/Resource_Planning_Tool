import os
from src.utils import COLOR_PALETTE

# 初始化分词器（使用常见的中文分词器）
class InitTokenizer:
    def __init__(self, model_path="./model/Qwen2___5-0___5B-Instruct"):
        """尝试加载分词器，如果失败则使用简单的字符计数"""
        self.model_path = model_path
        self.tokenizer, self.tokenizer_available = self.load_tokenizer(self.model_path)
    
    def load_tokenizer(self, model_path):
        try:
            # Lazy import to avoid hard dependency at server startup
            from transformers import AutoTokenizer  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer_available = True
        except Exception as e:
            print(f"警告: 无法加载分词器，将使用简单的字符计数: {e}")
            tokenizer = None
            tokenizer_available = False
        return tokenizer, tokenizer_available
    
    # 添加动态切换分词器方法
    def set_model_path(self, model_name):
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
        self.model_path = os.path.join(base_path, model_name)
        self.tokenizer, self.tokenizer_available = self.load_tokenizer(self.model_path)
        return self.tokenizer_available
    
    def count_tokens(self,text):
        """计算文本的token数量"""
        if not text or text.strip() == "":
            return 0
        
        if self.tokenizer_available and self.tokenizer:
            try:
                # 使用transformers分词器
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                print(f"分词器计算出错: {e}")
                # 降级到字符计数
                return len(text.replace(" ", "").replace("\n", ""))
        else:
            # 简单的字符计数（中文通常1个字符约等于1个token）
            return len(text.replace(" ", "").replace("\n", ""))
        
    def calculate_input_tokens(self,system_prompt, user_prompt):
        """计算系统提示词和用户输入的总token数，返回结构化JSON友好字典"""
        system_tokens = self.count_tokens(system_prompt) if system_prompt else 0
        user_tokens = self.count_tokens(user_prompt) if user_prompt else 0
        total_tokens = system_tokens + user_tokens

        system_chars = len(system_prompt) if system_prompt else 0
        user_chars = len(user_prompt) if user_prompt else 0
        total_chars = system_chars + user_chars

        return {
            "system_tokens": system_tokens,
            "user_tokens": user_tokens,
            "total_tokens": total_tokens,
            "system_chars": system_chars,
            "user_chars": user_chars,
            "total_chars": total_chars,
        }
    

class TokenProcessor:
    def __init__(self):
        self.examples=self.get_token_examples()

    def get_token_examples(self):
        """获取token计算示例"""
        examples = [
            {
                "name": "简单对话",
                "system": "你是一个有用的AI助手。",
                "user": "你好，请介绍一下自己。"
            },
            {
                "name": "代码助手",
                "system": "你是一个专业的Python编程助手，具备丰富的编程经验和深厚的计算机科学理论基础。你能够帮助用户解决各种编程问题，包括但不限于算法设计、数据结构、代码优化、调试技巧、最佳实践等。你会提供高质量的代码实现，并给出详细的解释和注释，确保用户能够理解代码逻辑和设计思路。",
                "user": "请帮我写一个Python函数来实现二分查找算法，要求能够在有序数组中查找指定元素并返回其索引。如果元素不存在则返回-1。请提供完整的代码实现，包括详细注释，并解释算法的时间复杂度和空间复杂度。"
            },
            {
                "name": "长文本分析",
                "system": "你是一个资深的文本分析专家和数据科学家，拥有丰富的自然语言处理经验。你擅长从大量复杂文本中提取关键信息、识别主要观点、分析情感倾向、发现潜在模式和趋势。你能够运用先进的文本挖掘技术，包括主题建模、情感分析、关键词提取、语义分析等方法，为用户提供深入的洞察和有价值的分析结果。你的分析既注重技术的严谨性，又兼顾实际应用的可操作性。",
                "user": """请深入分析以下关于人工智能发展的综合性报告，并从多个维度提供详细的分析结果：

**人工智能产业发展现状与未来展望报告**

随着深度学习、神经网络、大语言模型等核心技术的突破性进展，人工智能正在经历前所未有的快速发展期。从2020年到2024年，全球AI市场规模从约400亿美元增长到超过1500亿美元，预计到2030年将达到万亿美元级别。这一爆炸式增长主要得益于计算能力的大幅提升、数据资源的丰富积累，以及算法创新的持续突破。

在应用领域方面，AI技术已经深度渗透到社会生活的各个方面。在医疗健康领域，AI辅助诊断系统的准确率已达到甚至超越资深医生水平，特别是在影像识别、病理分析、药物研发等方面展现出巨大潜力。自动驾驶技术正逐步从L2级向L4、L5级发展，多家科技巨头和汽车制造商投入巨资推进相关研究。金融科技领域，智能风控、量化交易、个性化推荐等应用已成为行业标准。教育行业正在经历个性化学习革命，AI导师能够根据学生的学习特点和进度提供定制化的教学方案。

然而，AI的快速发展也带来了诸多挑战和争议。就业市场面临重大冲击，传统的重复性工作岗位正在被自动化替代，虽然同时也创造了新的就业机会，但技能转换和再就业问题日益突出。数据隐私和安全问题愈发严重，大规模数据收集和分析引发了公众对个人隐私泄露的担忧。算法偏见和歧视问题不容忽视，AI系统可能会放大现有的社会偏见，在招聘、信贷、司法等关键决策中产生不公平结果。此外，AI技术的军事化应用引发了国际社会对于"杀手机器人"和AI军备竞赛的担忧。

从技术发展趋势来看，多模态AI、联邦学习、边缘计算、量子计算与AI的融合等将成为下一阶段的重点发展方向。企业和研究机构正在加大对通用人工智能(AGI)的研发投入，试图创造出具有人类水平认知能力的AI系统。同时，AI的可解释性、鲁棒性、安全性等技术难题仍需要持续攻克。

监管层面，各国政府正在积极制定AI相关的法律法规和伦理准则。欧盟率先推出了《人工智能法案》，美国发布了AI权利法案，中国也在加快相关立法进程。国际合作与标准制定成为全球共识，但在具体实施细节上仍存在分歧。

产业生态方面，头部科技公司的竞争日趋激烈，算力、数据、人才成为核心竞争要素。开源与闭源模式并存，开源社区推动了技术的快速传播和迭代，而闭源模式则更注重商业化应用和盈利能力。投资热潮持续升温，风险投资、私募股权、政府基金等多方资本涌入AI领域，推动了技术创新和产业发展。

请从以下角度进行深度分析：1)技术发展趋势和突破点；2)各应用领域的影响和变革；3)社会经济影响评估；4)风险挑战和应对策略；5)未来发展前景预测；6)政策建议和行业建议。请提供具体的数据支撑和案例分析，确保分析的客观性和专业性。"""}

        ]
        return examples

    def load_example(self,example_name):
        """加载示例到输入框"""
        examples = self.examples
        for example in examples:
            if example["name"] == example_name:
                return example["system"], example["user"]
        return "", ""