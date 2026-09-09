import random
import re
from typing import List, Optional

class EnglishTextAugmenter:
    def __init__(self):
        # 增强概率参数
        self.case_prob = 0.3         # 大小写变换概率
        self.punctuation_prob = 0.2  # 标点符号变换概率
        self.stopword_remove_prob = 0.25  # 停用词删除概率
        self.synonym_prob = 0.3      # 同义词替换概率
        self.coordinate_swap_prob = 0.15  # 并列结构交换概率
        
        # 扩展的英文停用词表
        self.stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "in", "on", "at", "to", "of", "for", "with", "by", "from", "about",
            "and", "or", "but", "so", "yet", "nor", "as", "than", "then",
            "i", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them",
            "my", "your", "his", "her", "its", "our", "their",
            "this", "that", "these", "those",
            "which", "what", "who", "whom", "whose",
            "can", "could", "may", "might", "must", "shall", "should", "will", "would",
            "do", "does", "did", "have", "has", "had",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "too", "very"
        }
        
        # 扩展的英文同义词表（形容词、副词、部分动词）
        self.synonyms = {
            # 形容词
            "big": ["large", "huge", "vast", "massive", "gigantic"],
            "small": ["little", "tiny", "miniature", "petite", "diminutive"],
            "fast": ["quick", "swift", "rapid", "speedy", "hasty"],
            "slow": ["sluggish", "gradual", "leisurely", "tardy", "unhurried"],
            "happy": ["glad", "pleased", "joyful", "delighted", "content"],
            "sad": ["unhappy", "sorrowful", "melancholy", "gloomy", "depressed"],
            "good": ["excellent", "great", "fine", "superior", "admirable"],
            "bad": ["poor", "inferior", "terrible", "awful", "horrible"],
            "new": ["fresh", "recent", "modern", "novel", "contemporary"],
            "old": ["ancient", "vintage", "antique", "aged", "mature"],
            "strong": ["powerful", "mighty", "robust", "sturdy", "tough"],
            "weak": ["fragile", "feeble", "frail", "puny", "vulnerable"],
            
            # 副词
            "quickly": ["rapidly", "swiftly", "speedily", "promptly", "hastily"],
            "slowly": ["gradually", "sluggishly", "leisurely", "tardily", "unhurriedly"],
            "very": ["extremely", "highly", "greatly", "particularly", "exceptionally"],
            "often": ["frequently", "regularly", "habitually", "repeatedly", "commonly"],
            "rarely": ["seldom", "infrequently", "scarcely", "hardly", "barely"],
            "easily": ["effortlessly", "simply", "readily", "smoothly", "painlessly"],
            
            # 常用动词（选择不会改变核心语义的）
            "walk": ["stroll", "saunter", "wander", "hike", "trek"],
            "run": ["sprint", "jog", "dash", "race", "gallop"],
            "look": ["see", "watch", "view", "observe", "glance"],
            "say": ["speak", "tell", "state", "express", "utter"],
            "think": ["consider", "ponder", "reflect", "contemplate", "speculate"]
        }
        
        # 预编译正则表达式
        self.word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        self.punctuation_pattern = re.compile(r'[,.!?;:]')
        self.coordinate_pattern = re.compile(r'(\w+)( and | or )(\w+)')

    def random_case_change(self, text: str) -> str:
        """随机改变字母大小写"""
        result = []
        for word in text.split():
            if random.random() < self.case_prob:
                # 随机选择大小写变换方式
                if random.random() < 0.3:
                    # 首字母大写变小写
                    if word and word[0].isupper():
                        word = word[0].lower() + word[1:]
                elif random.random() < 0.6:
                    # 首字母小写变大写
                    if word and word[0].islower():
                        word = word[0].upper() + word[1:]
                else:
                    # 全词大小写反转
                    word = word.swapcase()
            result.append(word)
        return ' '.join(result)

    def random_punctuation_change(self, text: str) -> str:
        """随机变换标点符号"""
        if random.random() >= self.punctuation_prob:
            return text
            
        def replace_punct(match):
            punct = match.group(0)
            # 标点替换映射表
            replacements = {
                ',': [',', '', ';'],
                '.': ['.', '', '!', '?'],
                '!': ['!', '.', '?'],
                '?': ['?', '.', '!'],
                ';': [';', ',', ':'],
                ':': [':', ';', ',']
            }
            return random.choice(replacements.get(punct, [punct]))
            
        return self.punctuation_pattern.sub(replace_punct, text)

    def stopword_removal(self, text: str) -> str:
        """仅随机删除停用词，不添加新的停用词"""
        if random.random() >= self.stopword_remove_prob:
            return text
            
        words = text.split()
        new_words = []
        
        for word in words:
            # 随机删除停用词
            if word.lower() in self.stopwords and random.random() < 0.4:
                continue
                
            new_words.append(word)
                
        return ' '.join(new_words)

    def synonym_replacement(self, text: str) -> str:
        """同义词替换（仅使用自定义词表）"""
        if random.random() >= self.synonym_prob:
            return text
            
        words = text.split()
        
        for i, word in enumerate(words):
            # 跳过停用词
            if word.lower() in self.stopwords:
                continue
                
            # 从自定义词表获取同义词
            lower_word = word.lower()
            candidates = self.synonyms.get(lower_word, [])
            
            # 如果有可用的同义词，进行替换
            if candidates and random.random() < 0.4:
                replacement = random.choice(candidates)
                # 保持原词的大小写风格
                if word.isupper():
                    replacement = replacement.upper()
                elif word and word[0].isupper():
                    replacement = replacement.capitalize()
                words[i] = replacement
                
        return ' '.join(words)

    def coordinate_swap(self, text: str) -> str:
        """交换并列结构中的词语（如"cat and dog" <-> "dog and cat"）"""
        if random.random() < self.coordinate_swap_prob:
            def swap_match(match):
                return f"{match.group(3)}{match.group(2)}{match.group(1)}"
            return self.coordinate_pattern.sub(swap_match, text)
        return text

    def augment(self, text: str) -> str:
        """主增强函数，随机应用多种增强方法"""
        if not isinstance(text, str):
            return str(text)
            
        # 按随机顺序应用增强方法
        augmentations = [
            self.random_case_change,
            self.random_punctuation_change,
            self.stopword_removal,
            self.synonym_replacement,
            self.coordinate_swap
        ]
        
        # 随机打乱增强顺序并选择部分应用（1-3种）
        random.shuffle(augmentations)
        selected_augmentations = random.sample(augmentations, k=random.randint(1, 3))
        
        # 应用增强
        augmented_text = text
        for func in selected_augmentations:
            augmented_text = func(augmented_text)
            
        return augmented_text

    def augment_batch(self, texts: List[str], augment_prob: float = 0.6) -> List[str]:
        """批量增强文本，控制增强比例"""
        return [
            self.augment(text) if random.random() < augment_prob else text
            for text in texts
        ]

# 使用示例
if __name__ == "__main__":
    # 初始化增强器
    augmenter = EnglishTextAugmenter()
    
    # 测试文本
    test_texts = [
        "A black cat sitting on a red chair",
        "The quick brown fox jumps over the lazy dog",
        "A fast car and a slow truck",
        "She walks quickly through the park",
        "This is a good example of natural language processing"
    ]
    
    # 展示增强效果（每个文本生成3个增强版本）
    for text in test_texts:
        print(f"原始文本: {text}")
        for i in range(3):
            augmented = augmenter.augment(text)
            print(f"增强版本 {i+1}: {augmented}")
        print("---")
    