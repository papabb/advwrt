# -*- coding: utf-8 -*-
"""
Mock Fact-Check Server for Fast Training

This module provides a fast mock fact-check server that:
1. Knows the ground truth about what the Saboteur changed
2. Returns "FLAG" if detector checks a sabotaged claim
3. Returns "OK" if detector checks a non-sabotaged claim
4. Calculates F-1 score for tool usage accuracy
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict


class MockFactCheckServer:
    """
    Mock fact-check server that uses ground truth sabotage information.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset server state for a new example."""
        self.original_text = None
        self.sabotaged_text = None
        self.sabotage_type = None
        self.sabotage_changes = []  # List of changes made by saboteur
        self.sabotaged_claims = []  # Extracted claims that were sabotaged
        self.tool_calls = []  # Track all tool calls made
    
    def set_ground_truth(self, original_text: str, sabotaged_text: str, 
                         sabotage_type: str, changes: List[str]):
        """
        Set ground truth information about what was sabotaged.
        
        Args:
            original_text: Original text before sabotage
            sabotaged_text: Text after sabotage
            sabotage_type: Type of sabotage (grammar, style, clarity, logic, coherence)
            changes: List of changes made (from sabotage_writing function)
        """
        self.original_text = original_text
        self.sabotaged_text = sabotaged_text
        self.sabotage_type = sabotage_type
        self.sabotage_changes = changes
        
        # Extract sabotaged claims from changes
        self.sabotaged_claims = self._extract_claims_from_changes(changes, sabotaged_text)
    
    def _extract_claims_from_changes(self, changes: List[str], sabotaged_text: str) -> List[str]:
        """
        Extract specific claims that were sabotaged from the changes list.
        
        Args:
            changes: List of change descriptions
            sabotaged_text: The sabotaged text
        
        Returns:
            List of sabotaged claims/phrases
        """
        sabotaged_claims = []
        
        for change in changes:
            # Extract the actual changed text from change descriptions
            # Format examples:
            # "Grammar/typo: the -> teh"
            # "Wordiness: important -> very important and significant"
            # "Added vague phrase: and stuff like that"
            # "Added unsupported claim: This guarantees exceptional returns."
            
            if "->" in change:
                # Extract the "after" part (what it became)
                parts = change.split("->")
                if len(parts) == 2:
                    after = parts[1].strip()
                    # Find this in the sabotaged text
                    if after.lower() in sabotaged_text.lower():
                        sabotaged_claims.append(after)
            
            elif "Added" in change:
                # Extract what was added
                if ":" in change:
                    added_part = change.split(":", 1)[1].strip()
                    # Remove quotes if present
                    added_part = added_part.strip('"').strip("'")
                    if added_part:
                        sabotaged_claims.append(added_part)
        
        # Also extract sentences that contain the changes
        for change in changes:
            change_lower = change.lower()
            # Look for sentences in sabotaged text that might contain this change
            sentences = re.split(r'[.!?]\s+', sabotaged_text)
            for sentence in sentences:
                # Check if sentence contains keywords from change
                if any(word in sentence.lower() for word in change_lower.split() if len(word) > 3):
                    if sentence.strip() and sentence.strip() not in sabotaged_claims:
                        sabotaged_claims.append(sentence.strip())
        
        return sabotaged_claims
    
    def check_claim(self, claim: str) -> Dict[str, Any]:
        """
        Check a claim against ground truth sabotage information.
        
        Args:
            claim: The claim to check
        
        Returns:
            Dictionary with verdict and metadata
        """
        self.tool_calls.append(claim)
        
        # Check if claim matches any sabotaged content
        claim_lower = claim.lower()
        is_sabotaged = False
        
        # Check against sabotaged claims
        for sabotaged_claim in self.sabotaged_claims:
            if sabotaged_claim.lower() in claim_lower or claim_lower in sabotaged_claim.lower():
                is_sabotaged = True
                break
        
        # Check against sabotage changes
        for change in self.sabotage_changes:
            change_lower = change.lower()
            # Extract key phrases from change
            if "->" in change:
                after_part = change.split("->")[1].strip().lower()
                if after_part in claim_lower:
                    is_sabotaged = True
                    break
        
        # Check if claim contains sabotaged text snippets
        if self.sabotaged_text:
            # Look for sentences in sabotaged text that might be the claim
            sabotaged_sentences = re.split(r'[.!?]\s+', self.sabotaged_text)
            for sentence in sabotaged_sentences:
                sentence_lower = sentence.lower()
                # Check for significant overlap
                if len(claim_lower) > 10 and len(sentence_lower) > 10:
                    # Simple overlap check
                    claim_words = set(claim_lower.split())
                    sentence_words = set(sentence_lower.split())
                    overlap = len(claim_words & sentence_words)
                    if overlap > len(claim_words) * 0.5:  # 50% overlap
                        # Check if this sentence was modified
                        if sentence_lower not in (self.original_text or "").lower():
                            is_sabotaged = True
                            break
        
        # Return result
        if is_sabotaged:
            return {
                "claim": claim,
                "verdict": "FLAG",
                "confidence": 0.95,
                "explanation": f"This claim appears to be related to the sabotage ({self.sabotage_type}). The detector correctly identified a suspicious claim.",
                "evidence": [],
                "processing_time": 0.001,  # Fast!
                "timestamp": "",
                "domain": "mock"
            }
        else:
            return {
                "claim": claim,
                "verdict": "OK",
                "confidence": 0.9,
                "explanation": "This claim appears to be correct or unrelated to the sabotage.",
                "evidence": [],
                "processing_time": 0.001,  # Fast!
                "timestamp": "",
                "domain": "mock"
            }
    
    def calculate_f1_score(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate F-1 score for tool usage accuracy.
        
        Returns:
            Tuple of (f1_score, metrics_dict)
        """
        if not self.sabotaged_claims or len(self.sabotaged_claims) == 0:
            # No sabotage to detect - all tool calls are false positives
            if len(self.tool_calls) == 0:
                # No tool calls, no sabotage - True Negative
                return 1.0, {
                    "tp": 0, "fp": 0, "tn": 1, "fn": 0,
                    "precision": 1.0, "recall": 1.0, "f1": 1.0
                }
            else:
                # Tool calls but no sabotage - all False Positives
                return 0.0, {
                    "tp": 0, "fp": len(self.tool_calls), "tn": 0, "fn": 0,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0
                }
        
        # Count true positives, false positives, false negatives
        tp = 0  # Tool called for sabotaged claim
        fp = 0  # Tool called for non-sabotaged claim
        fn = 0  # Tool not called for sabotaged claim
        tn = 0  # Tool not called for non-sabotaged claim (hard to measure)
        
        # Check each tool call
        for tool_call in self.tool_calls:
            tool_call_lower = tool_call.lower()
            is_sabotaged = False
            
            # Check if tool call matches any sabotaged claim
            for sabotaged_claim in self.sabotaged_claims:
                if sabotaged_claim.lower() in tool_call_lower or tool_call_lower in sabotaged_claim.lower():
                    is_sabotaged = True
                    break
            
            # Also check against changes
            for change in self.sabotage_changes:
                if "->" in change:
                    after_part = change.split("->")[1].strip().lower()
                    if after_part in tool_call_lower:
                        is_sabotaged = True
                        break
            
            if is_sabotaged:
                tp += 1
            else:
                fp += 1
        
        # False negatives: sabotaged claims that weren't checked
        # We approximate this by checking if any sabotaged content wasn't queried
        checked_sabotages = set()
        for tool_call in self.tool_calls:
            tool_call_lower = tool_call.lower()
            for sabotaged_claim in self.sabotaged_claims:
                if sabotaged_claim.lower() in tool_call_lower or tool_call_lower in sabotaged_claim.lower():
                    checked_sabotages.add(sabotaged_claim.lower())
        
        fn = max(0, len(self.sabotaged_claims) - len(checked_sabotages))
        
        # Calculate precision, recall, F-1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        return f1, metrics


# Global mock server instance
_mock_server = MockFactCheckServer()


def call_mock_fact_check(claim: str) -> Dict[str, Any]:
    """
    Call the mock fact-check server.
    
    Args:
        claim: The claim to check
    
    Returns:
        Dictionary with fact-check results (same format as real API)
    """
    return _mock_server.check_claim(claim)


def set_mock_ground_truth(original_text: str, sabotaged_text: str, 
                         sabotage_type: str, changes: List[str]):
    """
    Set ground truth for the mock server.
    
    Args:
        original_text: Original text before sabotage
        sabotaged_text: Text after sabotage
        sabotage_type: Type of sabotage
        changes: List of changes made
    """
    _mock_server.set_ground_truth(original_text, sabotaged_text, sabotage_type, changes)


def get_mock_f1_score() -> Tuple[float, Dict[str, Any]]:
    """
    Get F-1 score for current example.
    
    Returns:
        Tuple of (f1_score, metrics_dict)
    """
    return _mock_server.calculate_f1_score()


def reset_mock_server():
    """Reset mock server for new example."""
    _mock_server.reset()
