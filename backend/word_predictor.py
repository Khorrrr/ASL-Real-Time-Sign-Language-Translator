"""
Word prediction using a HuggingFace language model.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


class WordPredictor:
    def __init__(self, model_name="gpt2"):
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models", "word_predictor", model_name
        )

        try:
            if os.path.exists(model_dir):
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                self.model = AutoModelForCausalLM.from_pretrained(model_dir)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.has_model = True
        except Exception as e:
            print(f"GPT-2 Load Error (Falling back to dictionary only): {e}")
            self.has_model = False

        self.common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", 
            "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", 
            "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", 
            "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", 
            "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", 
            "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", 
            "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", 
            "new", "want", "because", "any", "these", "give", "day", "most", "us", "hello", "help", "here", 
            "name", "nice", "please", "thank", "thanks", "thank you", "sorry", "need", "love", "happy", "feel", 
            "understand", "learn", "friend", "family", "home", "school", "water", "food", "yes", "where", 
            "why", "much", "many", "very", "been", "call", "more", "long", "made", "find", "down", "did", 
            "world", "again", "still", "hand", "high", "keep", "last", "let", "might", "never", "next", 
            "old", "right", "same", "small", "start", "tell", "turn", "every", "ask", "big", "end", "eye", 
            "far", "head", "leave", "life", "live", "move", "night", "off", "own", "play", "point", "read", 
            "real", "run", "seem", "show", "side", "talk", "walk", "watch", "always", "between", "city", 
            "close", "country", "enough", "important", "miss", "maybe", "really", "something", "sometimes",
            "working", "awesome", "great", "excellent", "nothing", "everything", "anyone", "someone",
            "how are you", "what's up", "good morning", "good night", "see you", "i love you",
            "able", "aboard", "about", "above", "accept", "accident", "according", "account", "accurate", "acres", "across", "act", "action", "active", "activity", 
            "actual", "actually", "add", "addition", "additional", "adjective", "adult", "adventure", "advice", "affect", "afraid", "after", "afternoon", "again", "against", "age", 
            "ago", "agree", "ahead", "aid", "air", "airplane", "alike", "alive", "all", "allow", "almost", "alone", "along", "aloud", "alphabet", "already", "also", 
            "although", "am", "among", "amount", "ancient", "angle", "angry", "animal", "announced", "another", "answer", "ants", "any", "anybody", "anyone", "anything", "anyway", 
            "anywhere", "apart", "apartment", "appearance", "apple", "applied", "appropriate", "are", "area", "arm", "army", "around", "arrange", "arrangement", "arrive", "arrow", 
            "art", "article", "as", "aside", "ask", "asleep", "aspect", "assemble", "at", "ate", "atmosphere", "atom", "atomic", "attached", "attack", "attempt", "attention", 
            "audience", "author", "automobile", "available", "average", "avoid", "aware", "away", "baby", "back", "bad", "badly", "bag", "balance", "ball", "balloon", "band", 
            "bank", "bar", "bare", "bark", "barn", "base", "baseball", "basic", "basis", "basket", "bat", "battle", "be", "bean", "bear", "beat", "beautiful", "beauty", 
            "became", "because", "become", "becoming", "bee", "been", "before", "began", "beginning", "begun", "behavior", "behind", "being", "believed", "bell", "belong", "below", 
            "belt", "bend", "beneath", "bent", "beside", "besides", "best", "bet", "better", "between", "beyond", "bicycle", "bigger", "biggest", "bill", "birds", "birth", 
            "birthday", "bit", "bite", "black", "blank", "blanket", "blew", "blind", "block", "blood", "blow", "blue", "board", "boat", "body", "bone", "book", "boots", 
            "born", "both", "bottle", "bottom", "bound", "bow", "bowl", "box", "boy", "brain", "branch", "brass", "brave", "bread", "break", "breakfast", "breath", 
            "breathe", "breathing", "breeze", "brick", "bridge", "brief", "bright", "bring", "broad", "broke", "broken", "brother", "brought", "brown", "brush", "buffalo", "build", 
            "building", "built", "buried", "burn", "burst", "bus", "bush", "business", "busy", "but", "butter", "buy", "by", "cabin", "cage", "cake", "call", "calm", 
            "came", "camera", "camp", "can", "canal", "can't", "cap", "capital", "captain", "captured", "car", "carbon", "card", "care", "careful", "carefully", "carried", 
            "carry", "case", "cast", "castle", "cat", "catch", "cattle", "caught", "cause", "cave", "cell", "cent", "center", "central", "century", "certain", "certainly", 
            "chain", "chair", "chamber", "chance", "change", "changing", "chapter", "character", "characteristic", "charge", "chart", "check", "cheese", "chemical", "chest", "chicken", "chief", 
            "child", "children", "choice", "choose", "chose", "chosen", "church", "circle", "circus", "citizen", "city", "class", "classroom", "claws", "clay", "clean", "clear", 
            "clearly", "climate", "climb", "clock", "close", "closely", "closer", "cloth", "clothes", "clothing", "cloud", "club", "coach", "coal", "coast", "coat", "coffee", 
            "cold", "collect", "college", "colony", "color", "column", "combination", "combine", "come", "comfortable", "coming", "command", "common", "community", "company", "compare", "compass", 
            "complete", "completely", "complex", "composed", "composition", "compound", "concerned", "condition", "congress", "connected", "consider", "consist", "constant", "constitution", "contain", "container", "content", 
            "continent", "continued", "contrast", "control", "conversation", "cook", "cookies", "cool", "copper", "copy", "corn", "corner", "correct", "correctly", "cost", "cotton", "could", 
            "count", "country", "couple", "courage", "course", "court", "cover", "cow", "cowboy", "crack", "cream", "create", "creature", "crew", "crop", "cross", "crowd", 
            "cry", "cup", "curious", "current", "curve", "customs", "cut", "cutting", "daily", "damage", "dance", "danger", "dangerous", "dark", "darkness", "date", "daughter", 
            "dawn", "day", "dead", "deal", "dear", "death", "decide", "declared", "deep", "deeply", "deer", "definition", "degree", "depend", "depth", "describe", "desert", 
            "design", "desk", "detail", "determine", "develop", "development", "diagram", "diameter", "did", "die", "differ", "difference", "different", "difficult", "difficulty", "dig", "dinner", 
            "direct", "direction", "directly", "dirt", "dirty", "disappear", "discover", "discovery", "discuss", "discussion", "disease", "dish", "distance", "distant", "divide", "division", "do", 
            "doctor", "does", "dog", "doing", "doll", "dollar", "done", "don't", "door", "dots", "double", "doubt", "down", "dozen", "draw", "drawn", "dream", 
            "dress", "drew", "dried", "drink", "drive", "driven", "driver", "driving", "drop", "dropped", "drove", "dry", "duck", "due", "dug", "dull", "during", 
            "dust", "duty", "each", "eager", "ear", "earlier", "early", "earn", "earth", "easier", "easily", "east", "easy", "eat", "eaten", "edge", "education", 
            "effect", "effort", "egg", "eight", "either", "electric", "electricity", "element", "elephant", "eleven", "else", "empty", "end", "enemy", "energy", "engine", "engineer", 
            "enjoy", "enough", "enter", "entire", "entirely", "environment", "equal", "equally", "equation", "equipment", "escape", "especially", "essential", "establish", "even", "evening", "event", 
            "ever", "every", "everybody", "everyone", "everything", "everywhere", "evidence", "exact", "exactly", "examine", "example", "excellent", "except", "exchange", "excited", "excitement", "exciting", 
            "exclaimed", "exercise", "exist", "expect", "experience", "experiment", "explain", "explanation", "explore", "express", "expression", "extra", "eye", "face", "fact", "factor", "factory", 
            "failed", "fair", "fairly", "fall", "fallen", "familiar", "family", "famous", "far", "farm", "farmer", "farther", "fast", "fastened", "faster", "fat", "father", 
            "favorite", "fear", "feather", "feature", "fed", "feed", "feel", "feet", "fell", "fellow", "felt", "fence", "few", "fewer", "field", "fierce", "fifteen", 
            "fifth", "fifty", "fight", "fighting", "figure", "fill", "film", "final", "finally", "find", "fine", "finest", "finger", "finish", "fire", "fireplace", "firm", 
            "first", "fish", "five", "fix", "flag", "flame", "flat", "flew", "flies", "flight", "floating", "floor", "flow", "flower", "fly", "fog", "folks", 
            "follow", "food", "foot", "football", "for", "force", "foreign", "forest", "forget", "forgot", "forgotten", "form", "former", "fort", "forth", "forty", "forward", 
            "fought", "found", "four", "fourth", "fox", "frame", "free", "freedom", "french", "frequent", "frequently", "fresh", "friend", "friendly", "frighten", "frog", "from", 
            "front", "frozen", "fruit", "fuel", "full", "fully", "fun", "function", "funny", "fur", "furniture", "further", "future", "gain", "game", "garage", "garden", 
            "gas", "gasoline", "gate", "gather", "gave", "general", "generally", "gentle", "gently", "get", "getting", "giant", "gift", "girl", "give", "given", "giving", 
            "glad", "glass", "glide", "glossary", "go", "goes", "gold", "golden", "gone", "good", "goose", "got", "government", "grabbed", "gradually", "grain", "graph", 
            "grass", "gravity", "gray", "great", "greater", "greatest", "greatly", "green", "grew", "ground", "group", "grow", "grown", "growth", "guard", "guess", "guide", 
            "gulf", "gun", "habit", "had", "hair", "half", "halfway", "hall", "hand", "handle", "handsome", "hang", "happen", "happened", "happily", "happy", "harbor", 
            "hard", "harder", "hardly", "has", "hat", "have", "having", "hay", "he", "head", "heading", "health", "heard", "heart", "heat", "heavy", "height", 
            "held", "hello", "help", "helpful", "her", "herd", "here", "herself", "hidden", "hide", "high", "higher", "highest", "highway", "hill", "him", "himself", 
            "his", "history", "hit", "hold", "hole", "hollow", "home", "honor", "hope", "horn", "horse", "hospital", "hot", "hour", "house", "how", "however", 
            "huge", "human", "hundred", "hung", "hungry", "hunt", "hunter", "hurried", "hurry", "hurt", "husband", "ice", "idea", "identity", "if", "ill", "image", 
            "imagine", "immediately", "importance", "important", "impossible", "improve", "in", "inch", "include", "including", "income", "increase", "indeed", "independent", "indicate", "individual", "industrial", 
            "industry", "influence", "information", "inside", "instance", "instant", "instead", "instrument", "interest", "interior", "into", "introduced", "invented", "iron", "is", "island", "it", 
            "its", "itself", "jack", "jar", "jet", "job", "join", "joined", "journey", "joy", "judge", "jump", "jungle", "just", "keep", "kept", "key", 
            "kids", "kill", "kind", "kitchen", "knew", "knife", "know", "knowledge", "known", "label", "labor", "lack", "lady", "laid", "lake", "lamp", "land", 
            "language", "large", "larger", "largest", "last", "late", "later", "laugh", "law", "lay", "layers", "lead", "leader", "leaf", "learn", "least", "leather", 
            "leave", "leaving", "led", "left", "leg", "length", "lesson", "let", "letter", "level", "library", "lie", "life", "lift", "light", "like", "likely", 
            "limited", "line", "lion", "lips", "liquid", "list", "listen", "little", "live", "living", "load", "local", "locate", "location", "log", "lonely", "long", 
            "longer", "look", "loose", "lose", "loss", "lost", "lot", "loud", "love", "lovely", "low", "lower", "luck", "lucky", "lunch", "lungs", "lying", 
            "machine", "machinery", "mad", "made", "magic", "magnet", "mail", "main", "mainly", "major", "make", "making", "man", "managed", "manner", "manufacturing", "many", 
            "map", "mark", "market", "married", "mass", "massage", "master", "material", "mathematics", "matter", "may", "maybe", "me", "meal", "mean", "means", "meant", 
            "measure", "meat", "medicine", "meet", "melted", "member", "memory", "men", "mental", "merely", "met", "metal", "method", "mice", "middle", "might", "mighty", 
            "mile", "military", "milk", "mill", "mind", "mine", "minerals", "minute", "mirror", "missing", "mission", "mistake", "mix", "mixture", "model", "modern", "molecular", 
            "moment", "money", "monkey", "month", "mood", "moon", "more", "morning", "most", "mostly", "mother", "motion", "mountain", "mouse", "mouth", "move", "movement", 
            "movie", "moving", "mud", "muscle", "music", "musical", "must", "my", "myself", "mysterious", "nails", "name", "nation", "national", "native", "natural", "naturally", 
            "nature", "near", "nearby", "nearer", "nearest", "nearly", "necessary", "neck", "needed", "needle", "needs", "negative", "neighbor", "neighborhood", "nervous", "nest", "never", 
            "new", "news", "newspaper", "next", "nice", "night", "nine", "no", "nobody", "nodded", "noise", "none", "noon", "nor", "north", "nose", "not", 
            "note", "noted", "nothing", "notice", "noun", "now", "number", "numeral", "nuts", "object", "observe", "obtain", "occasionally", "occur", "ocean", "of", "off", 
            "offer", "office", "officer", "official", "oil", "old", "older", "oldest", "on", "once", "one", "only", "onto", "open", "operation", "opinion", "opportunity", 
            "opposite", "or", "orange", "orbit", "order", "ordinary", "organization", "organized", "origin", "original", "other", "ought", "our", "ourselves", "out", "outer", "outline", 
            "outside", "over", "own", "owner", "oxygen", "pack", "package", "page", "paid", "pain", "paint", "pair", "palace", "pale", "pan", "paper", "paragraph", 
            "parallel", "parent", "park", "part", "particles", "particular", "particularly", "partly", "parts", "party", "pass", "passage", "past", "path", "pattern", "pay", "peace", 
            "pen", "pencil", "people", "per", "percent", "perfect", "perfectly", "perhaps", "period", "person", "personal", "phrase", "physical", "piano", "pick", "picture", "pictured", 
            "pie", "piece", "pig", "pile", "pilot", "pine", "pink", "pipe", "pitch", "place", "plain", "plan", "plane", "planet", "planned", "planning", "plant", 
            "plastic", "plate", "plates", "play", "pleasant", "please", "pleasure", "plenty", "plural", "plus", "pocket", "poem", "poet", "poetry", "point", "pole", "police", 
            "policeman", "political", "pond", "pony", "pool", "poor", "popular", "population", "port", "pose", "position", "positive", "possible", "possibly", "post", "pot", "potatoes", 
            "pound", "pour", "powder", "power", "powerful", "practical", "practice", "prepare", "present", "president", "press", "pressure", "pretty", "prevent", "previous", "price", "pride", 
            "primitive", "principal", "principle", "printed", "private", "prize", "probably", "problem", "process", "produce", "product", "production", "program", "progress", "promised", "proper", "properly", 
            "property", "protection", "proud", "prove", "provide", "public", "pull", "pupil", "pure", "purple", "purpose", "push", "put", "putting", "quarter", "queen", "question", 
            "quick", "quickly", "quiet", "quietly", "quite", "rabbit", "race", "radio", "railroad", "rain", "raise", "ran", "ranch", "range", "rapidly", "rate", "rather", 
            "raw", "rays", "reach", "read", "reader", "ready", "real", "realize", "rear", "reason", "recall", "receive", "recent", "recently", "recognize", "record", "red", 
            "refer", "refused", "region", "regular", "related", "relationship", "religious", "remain", "remarkable", "remember", "remove", "repeat", "replace", "replied", "report", "represent", "require", 
            "research", "respect", "rest", "result", "return", "review", "rhyme", "rhythm", "rice", "rich", "ride", "riding", "right", "ring", "rise", "rising", "river", 
            "road", "roar", "rock", "rocket", "rocky", "rod", "roll", "roof", "room", "root", "rope", "rose", "rough", "round", "route", "row", "rubbed", 
            "rubber", "rule", "ruler", "run", "running", "rush", "sad", "safe", "safety", "said", "sail", "sale", "salmon", "salt", "same", "sand", "sang", 
            "sat", "satellites", "satisfied", "save", "saved", "saw", "say", "scale", "scene", "school", "science", "scientific", "scientist", "score", "screen", "sea", "search", 
            "season", "seat", "second", "secret", "section", "see", "seed", "seeing", "seem", "seen", "seldom", "select", "selection", "sell", "send", "sense", "sent", 
            "sentence", "separate", "series", "serious", "serve", "service", "sets", "setting", "settle", "settlers", "seven", "several", "shade", "shadow", "shake", "shaking", "shall", 
            "shallow", "shape", "share", "sharp", "she", "sheep", "sheet", "shelf", "shell", "shelter", "shine", "shinning", "ship", "shirt", "shoe", "shook", "shoot", 
            "shore", "short", "shorter", "shot", "should", "shoulder", "shout", "show", "shown", "shut", "sick", "sides", "sight", "sign", "signal", "silence", "silent", 
            "silk", "silly", "silver", "similar", "simple", "simplest", "simply", "since", "sing", "single", "sink", "sister", "sit", "sitting", "situation", "six", "size", 
            "skill", "skin", "sky", "slabs", "slave", "sleep", "slept", "slide", "slight", "slightly", "slip", "slipped", "slope", "slow", "slowly", "small", "smaller", 
            "smallest", "smell", "smile", "smoke", "smooth", "snake", "snow", "so", "soap", "social", "society", "soft", "softly", "soil", "solar", "sold", "soldier", 
            "solid", "solution", "solve", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "son", "song", "soon", "sorry", "sort", "sound", 
            "source", "south", "southern", "space", "speak", "special", "species", "specific", "speech", "speed", "spell", "spend", "spent", "spider", "spin", "spirit", "spite", 
            "split", "spoken", "sport", "spread", "spring", "square", "stage", "stairs", "stand", "standard", "star", "stared", "start", "state", "statement", "station", "stay", 
            "steady", "steam", "steel", "steep", "stems", "step", "stepped", "stick", "stiff", "still", "stock", "stomach", "stone", "stood", "stop", "stopped", "store", 
            "storm", "story", "stove", "straight", "strange", "stranger", "straw", "stream", "street", "strength", "stretch", "strike", "string", "strip", "strong", "stronger", "struck", 
            "structure", "struggle", "stuck", "student", "studied", "study", "subject", "substance", "success", "successful", "such", "sudden", "suddenly", "sugar", "suggest", "suit", "sum", 
            "summer", "sun", "sunlight", "supper", "supply", "support", "suppose", "sure", "surface", "surprise", "surrounded", "swam", "sweet", "swept", "swim", "swimming", "swing", 
            "swung", "syllable", "symbol", "system", "table", "tail", "take", "taken", "tales", "talk", "tall", "tank", "tape", "task", "taste", "taught", "tax", 
            "tea", "teach", "teacher", "team", "tears", "teeth", "telephone", "television", "tell", "temperature", "ten", "tent", "term", "terrible", "test", "than", "thank", 
            "that", "thee", "them", "themselves", "then", "theory", "there", "therefore", "these", "they", "thick", "thin", "thing", "think", "third", "thirty", "this", 
            "those", "thou", "though", "thought", "thousand", "thread", "three", "threw", "throat", "through", "throughout", "throw", "thrown", "thumb", "thus", "thy", "tide", 
            "tie", "tight", "tightly", "till", "time", "tin", "tiny", "tip", "tired", "title", "to", "tobacco", "today", "together", "told", "tomorrow", "tone", 
            "tongue", "tonight", "too", "took", "tool", "top", "topic", "torn", "total", "touch", "toward", "tower", "town", "toy", "trace", "track", "trade", 
            "traffic", "trail", "train", "transportation", "trap", "travel", "treated", "tree", "triangle", "tribe", "trick", "tried", "trip", "troops", "tropical", "trouble", "truck", 
            "trunk", "truth", "try", "tube", "tune", "turn", "twelve", "twenty", "twice", "two", "type", "typical", "uncle", "under", "underline", "understanding", "unhappy", 
            "union", "unit", "universe", "unknown", "unless", "until", "unusual", "up", "upon", "upper", "upward", "us", "use", "useful", "using", "usual", "usually", 
            "valley", "valuable", "value", "vapor", "variety", "various", "vast", "vegetable", "verb", "vertical", "very", "vessels", "victory", "view", "village", "visit", "visitor", 
            "voice", "volume", "vote", "vowel", "voyage", "wagon", "wait", "walk", "wall", "want", "war", "warm", "warn", "was", "wash", "waste", "watch", 
            "water", "wave", "way", "we", "weak", "wealth", "wear", "weather", "week", "weigh", "weight", "welcome", "well", "went", "were", "west", "western", 
            "wet", "whale", "what", "whatever", "wheat", "wheel", "when", "whenever", "where", "wherever", "whether", "which", "while", "whispered", "whistle", "white", "who", 
            "whole", "whom", "whose", "why", "wide", "widely", "wife", "wild", "will", "willing", "win", "wind", "window", "wing", "winter", "wire", "wise", 
            "wish", "with", "within", "without", "wolf", "women", "won", "wonder", "wonderful", "wood", "wooden", "wool", "word", "wore", "work", "worker", "working", 
            "world", "worried", "worry", "worse", "worth", "would", "wrapped", "write", "writer", "writing", "written", "wrong", "wrote", "yard", "year", "yellow", "yes", 
            "yesterday", "yet", "you", "young", "younger", "your", "yourself", "youth", "zero", "zebra", "zipper", "zoo", "zulu"
        ]

    def get_suggestions(self, sentence_so_far, current_letters, top_k=5):
        """
        Hybrid logic:
        1. If current_letters exists -> Dictionary Prefix Completion.
        2. If current_letters is empty -> GPT-2 Next Word Prediction.
        """
        prefix = current_letters.lower().strip()
        sentence = sentence_so_far.strip()

        if prefix:
            candidates = [w for w in self.common_words if w.startswith(prefix)]
            
            # If dictionary is weak, use GPT-2 to complete the word
            if len(candidates) < top_k and self.has_model:
                lm_completions = self._lm_complete_word(sentence, prefix, top_k - len(candidates))
                candidates.extend(lm_completions)
            
            suggestions = []
            seen = set()
            for w in candidates:
                w = w.lower()
                if w not in seen and w != prefix:
                    suggestions.append(w)
                    seen.add(w)
            return suggestions[:top_k]
        
        if not sentence:
            return ["i", "hello", "the", "how", "what"][:top_k]
        
        if self.has_model:
            return self._lm_next_word(sentence, top_k)
        else:
            return ["am", "is", "and", "the", "you"][:top_k]

    def _lm_complete_word(self, sentence, prefix, count):
        """Uses GPT-2 to complete a partially typed word."""
        try:
            prompt = f"{sentence} {prefix}" if sentence else prefix
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs, max_new_tokens=3, num_return_sequences=count,
                do_sample=True, temperature=0.8, pad_token_id=self.tokenizer.eos_token_id
            )
            
            results = []
            for out in outputs:
                text = self.tokenizer.decode(out, skip_special_tokens=True)
                after_prompt = text[len(prompt):].strip()
                word_match = re.match(r"^[a-zA-Z']+", after_prompt)
                if word_match:
                    results.append(prefix + word_match.group(0).lower())
            return results
        except: return []

    def _lm_next_word(self, sentence, count):
        """Uses GPT-2 to predict the entire next word based on context."""
        try:
            inputs = self.tokenizer(sentence, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=5, num_return_sequences=count,
                    do_sample=True, temperature=0.9, top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            results = []
            for out in outputs:
                text = self.tokenizer.decode(out, skip_special_tokens=True)
                next_part = text[len(sentence):].strip()
                first_word = next_part.split()[0] if next_part.split() else ""
                clean_word = re.sub(r'[^a-zA-Z\']', '', first_word).lower()
                if clean_word: results.append(clean_word)
            return results
        except: return []
