TRIPLE_INSTRUCTIONS = {
    "en":{
        "system": "You are a helpful assistant who always response in a valid array of JSON objects without any explanation",
        "entity_relation": """Given a passage, summarize all the important entities and the relations between them in a concise manner. Relations should briefly capture the connections between entities, without repeating information from the head and tail entities. The entities should be as specific as possible. Exclude pronouns from being considered as entities. 
        You must **strictly output in the following JSON format**:\n
        [
            {
                "Head": "{a noun}",
                "Relation": "{a verb}",
                "Tail": "{a noun}",
            }...
        ]
        Here is the passage:""",

        "event_entity":  """Please analyze and summarize the participation relations between the events and entities in the given paragraph. Each event is a single independent sentence. Additionally, identify all the entities that participated in the events. Do not use ellipses. 
        You must **strictly output in the following JSON format**:\n
        [
            {
                "Event": "{a simple sentence describing an event}",
                "Entity": ["entity 1", "entity 2", "..."]
            }...
        ]
        Here is the passage:""",
        "event_relation":  """Please analyze and summarize the relationships between the events in the paragraph. Each event is a single independent sentence. Identify temporal and causal relationships between the events using the following types: before, after, at the same time, because, and as a result. Each extracted triple should be specific, meaningful, and able to stand alone.  Do not use ellipses.  
        You must **strictly output in the following JSON format**:\n
        [
            {
                "Head": "{a simple sentence describing the event 1}",
                "Relation": "{temporal or causality relation between the events}",
                "Tail": "{a simple sentence describing the event 2}"
            }...
        ]
        Here is the passage:""",
    },
    "zh-CN": {
        "system": """"你是一个始终以有效JSON数组格式回应的助手""",
        "entity_relation": """给定一段文字，提取所有重要实体及其关系，并以简洁的方式总结。关系描述应清晰表达实体间的联系，且不重复头尾实体的信息。实体需具体明确，排除代词。  
        返回格式必须为以下JSON结构,内容需用简体中文表述:
        [  
            {  
                "Head": "{名词}",  
                "Relation": "{动词或关系描述}",  
                "Tail": "{名词}"  
            }...  
        ]
        给定以下段落：""",

        "event_entity": """分析段落中的事件及其参与实体。每个事件应为独立单句，列出所有相关实体（需具体，不含代词）。  
        返回格式必须为以下JSON结构,内容需用简体中文表述:
        [  
            {  
                "Event": "{描述事件的简单句子}",  
                "Entity": ["实体1", "实体2", "..."]  
            }...  
        ]
        给定以下段落：""",
       
        "event_relation": """分析事件间的时序或因果关系,关系类型包括:之前,之后,同时,因为,结果.每个事件应为独立单句。  
        返回格式必须为以下JSON结构.内容需用简体中文表述. 
        [  
            {  
                "Head": "{事件1描述}",  
                "Relation": "{时序/因果关系}",  
                "Tail": "{事件2描述}"  
            }...  
        ]
        给定以下段落：""",
    },
    "zh-HK": {
        "system": "你是一個始終以有效JSON數組格式回覆的助手",
        "entity_relation": """給定一段文字，提取所有重要實體及其關係，並以簡潔的方式總結。關係描述應清晰表達實體間的聯繫，且不重複頭尾實體的信息。實體需具體明確，排除代詞。  
        返回格式必須為以下JSON結構,內容需用繁體中文表述:
        [  
            {  
                "Head": "{名詞}",  
                "Relation": "{動詞或關係描述}",  
                "Tail": "{名詞}"  
            }...  
        ]
        給定以下段落：""",

        "event_entity": """分析段落中的事件及其參與實體。每個事件應為獨立單句，列出所有相關實體（需具體，不含代詞）。  
        返回格式必須為以下JSON結構,內容需用繁體中文表述:
        [  
            {  
                "Event": "{描述事件的簡單句子}",  
                "Entity": ["實體1", "實體2", "..."]  
            }...  
        ]
        給定以下段落：""",
       
        "event_relation": """分析事件間的時序或因果關係,關係類型包括:之前,之後,同時,因為,結果.每個事件應為獨立單句。  
        返回格式必須為以下JSON結構.內容需用繁體中文表述. 
        [  
            {  
                "Head": "{事件1描述}",  
                "Relation": "{時序/因果關係}",  
                "Tail": "{事件2描述}"  
            }...  
        ]
        給定以下段落：""",
    }
}

CONCEPT_INSTRUCTIONS = {
    "en": {
        "event": """I will give you an EVENT. You need to give several phrases containing 1-2 words for the ABSTRACT EVENT of this EVENT.
            You must return your answer in the following format: phrases1, phrases2, phrases3,...
            You can't return anything other than answers.
            These abstract event words should fulfill the following requirements.
            1. The ABSTRACT EVENT phrases can well represent the EVENT, and it could be the type of the EVENT or the related concepts of the EVENT.    
            2. Strictly follow the provided format, do not add extra characters or words.
            3. Write at least 3 or more phrases at different abstract level if possible.
            4. Do not repeat the same word and the input in the answer.
            5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

            EVENT: A man retreats to mountains and forests.
            Your answer: retreat, relaxation, escape, nature, solitude
            EVENT: A cat chased a prey into its shelter
            Your answer: hunting, escape, predation, hidding, stalking
            EVENT: Sam playing with his dog
            Your answer: relaxing event, petting, playing, bonding, friendship
            EVENT: [EVENT]
            Your answer:""",
        "entity":"""I will give you an ENTITY. You need to give several phrases containing 1-2 words for the ABSTRACT ENTITY of this ENTITY.
            You must return your answer in the following format: phrases1, phrases2, phrases3,...
            You can't return anything other than answers.
            These abstract intention words should fulfill the following requirements.
            1. The ABSTRACT ENTITY phrases can well represent the ENTITY, and it could be the type of the ENTITY or the related concepts of the ENTITY.
            2. Strictly follow the provided format, do not add extra characters or words.
            3. Write at least 3 or more phrases at different abstract level if possible.
            4. Do not repeat the same word and the input in the answer.
            5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

            ENTITY: Soul
            CONTEXT: premiered BFI London Film Festival, became highest-grossing Pixar release
            Your answer: movie, film

            ENTITY: Thinkpad X60
            CONTEXT: Richard Stallman announced he is using Trisquel on a Thinkpad X60
            Your answer: Thinkpad, laptop, machine, device, hardware, computer, brand

            ENTITY: Harry Callahan
            CONTEXT: bluffs another robber, tortures Scorpio
            Your answer: person, Amarican, character, police officer, detective

            ENTITY: Black Mountain College
            CONTEXT: was started by John Andrew Rice, attracted faculty
            Your answer: college, university, school, liberal arts college

            EVENT: 1st April
            CONTEXT: Utkal Dibas celebrates
            Your answer: date, day, time, festival

            ENTITY: [ENTITY]
            CONTEXT: [CONTEXT]
            Your answer:""",
        "relation":"""I will give you an RELATION. You need to give several phrases containing 1-2 words for the ABSTRACT RELATION of this RELATION.
            You must return your answer in the following format: phrases1, phrases2, phrases3,...
            You can't return anything other than answers.
            These abstract intention words should fulfill the following requirements.
            1. The ABSTRACT RELATION phrases can well represent the RELATION, and it could be the type of the RELATION or the simplest concepts of the RELATION.
            2. Strictly follow the provided format, do not add extra characters or words.
            3. Write at least 3 or more phrases at different abstract level if possible.
            4. Do not repeat the same word and the input in the answer.
            5. Stop immediately if you can't think of any more phrases, and no explanation is needed.
            
            RELATION: participated in
            Your answer: become part of, attend, take part in, engage in, involve in
            RELATION: be included in
            Your answer: join, be a part of, be a member of, be a component of
            RELATION: [RELATION]
            Your answer:"""
    },
    "zh-CN": {
         "event": """我将给你一个事件。你需要为这个事件的抽象概念提供几个1-2个词的短语。
            你必须按照以下格式返回答案：短语1, 短语2, 短语3,...
            除了答案外不要返回任何其他内容，请以简体中文输出。
            这些抽象事件短语应满足以下要求：
            1. 能很好地代表该事件的类型或相关概念
            2. 严格遵循给定格式，不要添加额外字符或词语
            3. 尽可能提供3个或以上不同抽象层次的短语
            4. 不要重复相同词语或输入内容
            5. 如果无法想出更多短语立即停止，不需要解释

            事件：一个人退隐到山林中
            你的回答：退隐, 放松, 逃避, 自然, 独处
            事件：一只猫将猎物追进巢穴
            你的回答：捕猎, 逃跑, 捕食, 躲藏, 潜行
            事件：山姆和他的狗玩耍
            你的回答：休闲活动, 抚摸, 玩耍, bonding, 友谊
            事件：[EVENT]
            请以简体中文输出你的回答：""",
        "entity":"""我将给你一个实体。你需要为这个实体的抽象概念提供几个1-2个词的短语。
            你必须按照以下格式返回答案：短语1, 短语2, 短语3,...
            除了答案外不要返回任何其他内容，请以简体中文输出。
            这些抽象实体短语应满足以下要求：
            1. 能很好地代表该实体的类型或相关概念
            2. 严格遵循给定格式，不要添加额外字符或词语
            3. 尽可能提供3个或以上不同抽象层次的短语
            4. 不要重复相同词语或输入内容
            5. 如果无法想出更多短语立即停止，不需要解释

            实体：心灵奇旅
            上下文：在BFI伦敦电影节首映，成为皮克斯最卖座影片
            你的回答：电影, 影片
            实体：Thinkpad X60
            上下文：Richard Stallman宣布他在Thinkpad X60上使用Trisquel系统
            你的回答：Thinkpad, 笔记本电脑, 机器, 设备, 硬件, 电脑, 品牌
            实体：哈利·卡拉汉
            上下文：吓退另一个劫匪，折磨天蝎座
            你的回答：人物, 美国人, 角色, 警察, 侦探
            实体：黑山学院
            上下文：由John Andrew Rice创办，吸引了众多教员
            你的回答：学院, 大学, 学校, 文理学院
            事件：4月1日
            上下文：庆祝Utkal Dibas
            你的回答：日期, 日子, 时间, 节日
            实体：[ENTITY]
            上下文：[CONTEXT]
            请以简体中文输出你的回答：""",
        "relation":"""我将给你一个关系。你需要为这个关系的抽象概念提供几个1-2个词的短语。
            你必须按照以下格式返回答案：短语1, 短语2, 短语3,...
            除了答案外不要返回任何其他内容，请以简体中文输出。
            这些抽象关系短语应满足以下要求：
            1. 能很好地代表该关系的类型或最基本概念
            2. 严格遵循给定格式，不要添加额外字符或词语
            3. 尽可能提供3个或以上不同抽象层次的短语
            4. 不要重复相同词语或输入内容
            5. 如果无法想出更多短语立即停止，不需要解释
            
            关系：参与
            你的回答：成为一部分, 参加, 参与其中, 涉及, 卷入
            关系：被包含在
            你的回答：加入, 成为一部分, 成为成员, 成为组成部分
            关系：[RELATION]
            请以简体中文输出你的回答："""
    },
    "zh-HK": {
         "event": """我將給你一個事件。你需要為這個事件的抽象概念提供幾個1-2個詞的短語。
            你必須按照以下格式返回答案：短語1, 短語2, 短語3,...
            除了答案外不要返回任何其他內容，請以繁體中文輸出。
            這些抽象事件短語應滿足以下要求：
            1. 能很好地代表該事件的類型或相關概念
            2. 嚴格遵循給定格式，不要添加額外字符或詞語
            3. 盡可能提供3個或以上不同抽象層次的短語
            4. 不要重複相同詞語或輸入內容
            5. 如果無法想出更多短語立即停止，不需要解釋

            事件：一個人退隱到山林中
            你的回答：退隱, 放鬆, 逃避, 自然, 獨處
            事件：一隻貓將獵物追進巢穴
            你的回答：捕獵, 逃跑, 捕食, 躲藏, 潛行
            事件：山姆和他的狗玩耍
            你的回答：休閒活動, 撫摸, 玩耍, bonding, 友誼
            事件：[EVENT]
            請以繁體中文輸出你的回答：""",
        "entity":"""我將給你一個實體。你需要為這個實體的抽象概念提供幾個1-2個詞的短語。
            你必須按照以下格式返回答案：短語1, 短語2, 短語3,...
            除了答案外不要返回任何其他內容，請以繁體中文輸出。
            這些抽象實體短語應滿足以下要求：
            1. 能很好地代表該實體的類型或相關概念
            2. 嚴格遵循給定格式，不要添加額外字符或詞語
            3. 盡可能提供3個或以上不同抽象層次的短語
            4. 不要重複相同詞語或輸入內容
            5. 如果無法想出更多短語立即停止，不需要解釋

            實體：心靈奇旅
            上下文：在BFI倫敦電影節首映，成為皮克斯最賣座影片
            你的回答：電影, 影片
            實體：Thinkpad X60
            上下文：Richard Stallman宣布他在Thinkpad X60上使用Trisquel系統
            你的回答：Thinkpad, 筆記本電腦, 機器, 設備, 硬件, 電腦, 品牌
            實體：哈利·卡拉漢
            上下文：嚇退另一個劫匪，折磨天蠍座
            你的回答：人物, 美國人, 角色, 警察, 偵探
            實體：黑山學院
            上下文：由John Andrew Rice創辦，吸引了眾多教員
            你的回答：學院, 大學, 學校, 文理學院
            事件：4月1日
            上下文：慶祝Utkal Dibas
            你的回答：日期, 日子, 時間, 節日
            實體：[ENTITY]
            上下文：[CONTEXT]
            請以繁體中文輸出你的回答：""",
        "relation":"""我將給你一個關係。你需要為這個關係的抽象概念提供幾個1-2個詞的短語。
            你必須按照以下格式返回答案：短語1, 短語2, 短語3,...
            除了答案外不要返回任何其他內容，請以繁體中文輸出。
            這些抽象關係短語應滿足以下要求：
            1. 能很好地代表該關係的類型或最基本概念
            2. 嚴格遵循給定格式，不要添加額外字符或詞語
            3. 盡可能提供3個或以上不同抽象層次的短語
            4. 不要重複相同詞語或輸入內容
            5. 如果無法想出更多短語立即停止，不需要解釋
            
            關係：參與
            你的回答：成為一部分, 參加, 參與其中, 涉及, 捲入
            關係：被包含在
            你的回答：加入, 成為一部分, 成為成員, 成為組成部分
            關係：[RELATION]
            請以繁體中文輸出你的回答："""
    }
}
