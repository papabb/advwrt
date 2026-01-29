# -*- coding: utf-8 -*-
"""
Data loading and preprocessing for Writing Quality Adversarial GRPO Training

This module handles:
- Loading writing datasets from various sources
- Generating synthetic writing samples as fallback
- Sabotage functions for introducing writing quality issues
"""

import os
import re
import random
from datasets import load_dataset, Dataset


def load_writing_dataset():
    """
    Load general writing dataset for criticism/adversarial training.
    Tries multiple sources, fallback to synthetic data if needed.
    """
    print("Loading writing dataset...")
    
    # Try to load from HuggingFace datasets (general writing)
    dataset_sources = [
        # Essay/story datasets
        ("jonathanli/human-essays-reddit", None, "train", ["top_comment", "selftext", "title"]),
        # Common text datasets
        ("wikitext", "wikitext-2-raw-v1", "train", ["text"]),
        ("bookcorpus", None, "train", ["text"]),
        # Writing/review datasets
        ("imdb", None, "train", ["text", "review"]),
        # Try simpler text datasets
        ("squad", None, "train", ["context", "question"]),
    ]
    
    for dataset_name, config_name, split_name, text_fields in dataset_sources:
        try:
            # Load dataset
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split=f"{split_name}[:1000]")
            else:
                dataset = load_dataset(dataset_name, split=f"{split_name}[:1000]")
            
            # Find text field
            text_field = None
            for field in text_fields:
                if field in dataset.column_names:
                    text_field = field
                    break
            
            if text_field:
                # Filter out empty or very short texts
                dataset = dataset.filter(lambda x: x.get(text_field) and len(str(x[text_field]).strip()) > 200)
                if len(dataset) > 0:
                    # Rename to 'text' for consistency
                    if text_field != "text":
                        dataset = dataset.rename_column(text_field, "text")
                    print(f"✓ Loaded {dataset_name} dataset: {len(dataset)} examples")
                    return dataset
        except Exception as e:
            print(f"  ✗ Failed to load {dataset_name}: {e}")
            continue
    
    # Fallback: Create synthetic writing samples (essays, stories, articles)
    print("Creating synthetic writing dataset...")
    synthetic_writings = []
    
    writing_types = [
        "essay", "story", "article", "review", "blog_post", "analysis"
    ]
    
    topics = [
        "climate change", "technology", "education", "health", "society",
        "science", "history", "culture", "politics", "economics"
    ]
    
    for i in range(500):
        writing_type = random.choice(writing_types)
        topic = random.choice(topics)
        
        # Generate synthetic writing sample
        if writing_type == "essay":
            text = f"""
Essay on {topic.title()}

{topic.title()} is a topic of great importance in today's world. It affects many aspects of our daily lives and requires careful consideration.

First, we must understand the fundamental principles underlying {topic}. This involves examining various perspectives and analyzing the available evidence. Research has shown that {topic} plays a crucial role in shaping our understanding of the world.

Furthermore, the implications of {topic} extend beyond immediate concerns. They influence long-term outcomes and require strategic thinking. It is essential to consider multiple viewpoints when discussing this subject.

In conclusion, {topic} represents a complex issue that demands thoughtful analysis and informed discussion. By engaging with this topic seriously, we can develop better insights and make more informed decisions.
"""
        elif writing_type == "story":
            characters = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
            character = random.choice(characters)
            text = f"""
A Story About {character}

{character} walked through the quiet streets, thinking about {topic}. The evening air was crisp, and the stars were beginning to appear in the sky.

As {character} continued walking, memories of past experiences with {topic} came flooding back. Each memory carried its own weight, its own lesson learned.

The journey ahead would not be easy, but {character} knew that understanding {topic} was essential. With determination and an open mind, the path forward became clearer.

In the end, {character} realized that {topic} was not just a concept to be studied, but a living part of everyday experience that shaped who we become.
"""
        elif writing_type == "article":
            text = f"""
Article: Understanding {topic.title()}

Recent developments in the field of {topic} have sparked renewed interest among researchers and practitioners. This article explores the key aspects and implications.

The current state of {topic} reflects broader trends in society. Experts suggest that we are witnessing a significant shift in how we approach this subject. Data from multiple studies indicate important patterns worth examining.

Key findings reveal that {topic} influences decision-making processes in meaningful ways. Stakeholders across various sectors are taking note of these developments and adjusting their strategies accordingly.

Looking ahead, the future of {topic} appears to be shaped by ongoing research and practical applications. Continued engagement with this topic will likely yield valuable insights.
"""
        else:
            # Generic writing
            text = f"""
Writing on {topic.title()}

{topic.title()} is a subject that deserves careful attention. This piece explores various dimensions of the topic and considers its broader significance.

The complexity of {topic} requires us to look beyond surface-level observations. By delving deeper, we can uncover important connections and patterns that might otherwise go unnoticed.

Practical applications of understanding {topic} are numerous. From individual decision-making to broader policy considerations, the implications are far-reaching.

Ultimately, engaging thoughtfully with {topic} helps us navigate the challenges and opportunities that lie ahead. It is through such engagement that we can hope to make meaningful progress.
"""
        
        synthetic_writings.append({
            "text": text.strip(),
            "type": writing_type,
            "topic": topic,
        })
    
    return Dataset.from_list(synthetic_writings)


def sabotage_writing(text, sabotage_type="mixed"):
    """
    Apply various sabotage techniques to degrade writing quality.
    Returns the sabotaged text and metadata about what was changed.
    """
    sabotaged = text
    changes = []
    
    if sabotage_type == "grammar" or sabotage_type == "mixed":
        # Introduce grammatical errors and typos
        grammar_errors = [
            ("the", "teh"),  # Common typo
            ("and", "adn"),  # Typo
            ("because", "becuase"),  # Typo
            ("their", "there"),  # Homophone error
            ("its", "it's"),  # Apostrophe error (wrong context)
            ("affect", "effect"),  # Common confusion
            ("you're", "your"),  # Apostrophe error
            ("they're", "their"),  # Homophone error
        ]
        if random.random() < 0.4:
            old, new = random.choice(grammar_errors)
            if old in sabotaged.lower():
                # Replace case-insensitively but preserve case
                pattern = re.compile(re.escape(old), re.IGNORECASE)
                sabotaged = pattern.sub(new, sabotaged, count=1)
                changes.append(f"Grammar/typo: {old} -> {new}")
    
    if sabotage_type == "style" or sabotage_type == "mixed":
        # Introduce awkward phrasing and wordiness
        style_issues = [
            ("important", "very important and significant"),
            ("because", "due to the fact that"),
            ("use", "utilize"),
            ("help", "assist in helping"),
            ("many", "a large number of"),
            ("often", "on a frequent basis"),
            ("now", "at this point in time"),
        ]
        if random.random() < 0.3:
            old, new = random.choice(style_issues)
            if old in sabotaged.lower():
                import re
                pattern = re.compile(re.escape(old), re.IGNORECASE)
                sabotaged = pattern.sub(new, sabotaged, count=1)
                changes.append(f"Wordiness: {old} -> {new}")
    
    if sabotage_type == "clarity" or sabotage_type == "mixed":
        # Add vague or unclear phrases
        vague_phrases = [
            "and stuff like that",
            "or something",
            "kind of",
            "sort of",
            "you know",
            "basically",
            "pretty much",
        ]
        if random.random() < 0.3:
            phrase = random.choice(vague_phrases)
            sentences = sabotaged.split(".")
            if len(sentences) > 1:
                insert_pos = random.randint(0, len(sentences) - 1)
                if sentences[insert_pos].strip():
                    sentences[insert_pos] = sentences[insert_pos] + f" {phrase}."
                    sabotaged = ".".join(sentences)
                    changes.append(f"Added vague phrase: {phrase}")
    
    if sabotage_type == "logic" or sabotage_type == "mixed":
        # Add logical inconsistencies or unsupported claims
        unsupported_claims = [
            "This is definitely true.",
            "Everyone knows this.",
            "It's obvious that",
            "This always happens.",
            "This never fails.",
        ]
        if random.random() < 0.25:
            claim = random.choice(unsupported_claims)
            sentences = sabotaged.split(".")
            if len(sentences) > 1:
                insert_pos = random.randint(0, len(sentences) - 1)
                if sentences[insert_pos].strip():
                    sentences[insert_pos] = f"{claim} {sentences[insert_pos]}"
                    sabotaged = ".".join(sentences)
                    changes.append(f"Added unsupported claim: {claim}")
    
    if sabotage_type == "coherence" or sabotage_type == "mixed":
        # Break coherence by adding irrelevant sentences
        if random.random() < 0.2:
            irrelevant_sentences = [
                "This reminds me of something else entirely.",
                "On a completely different note,",
                "But that's a topic for another time.",
            ]
            sentence = random.choice(irrelevant_sentences)
            sentences = sabotaged.split(".")
            if len(sentences) > 2:
                insert_pos = random.randint(1, len(sentences) - 1)
                sentences.insert(insert_pos, sentence)
                sabotaged = ".".join(sentences)
                changes.append(f"Added irrelevant sentence: {sentence}")
    
    return sabotaged, changes
