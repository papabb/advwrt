# -*- coding: utf-8 -*-
"""
Fact-Check Tool Integration Module

This module handles interactions with the fact-check API at http://ml18-pc08:8884/
"""

import requests
import json
import re
from typing import Optional, Dict, Any, List

# API Configuration
FACT_CHECK_API_URL = "http://ml18-pc08:8884/fact_check"
FACT_CHECK_TIMEOUT = 60  # API takes ~15-20 seconds, use 60s timeout
FACT_CHECK_MAX_CALLS_PER_DETECTION = 3  # Limit tool calls per text

# Tool calling tokens
TOOL_CALL_START = "<TOOL_CALL_START>"
TOOL_CALL_END = "<TOOL_CALL_END>"
TOOL_QUERY_START = "<TOOL_QUERY>"
TOOL_QUERY_END = "</TOOL_QUERY>"
TOOL_RESULT_START = "<TOOL_RESULT>"
TOOL_RESULT_END = "</TOOL_RESULT>"


def call_fact_check(claim: str, timeout: int = FACT_CHECK_TIMEOUT) -> Dict[str, Any]:
    """
    Call the fact-check API with a claim.
    
    Args:
        claim: The claim or statement to verify
        timeout: Request timeout in seconds (default: 60)
    
    Returns:
        Dictionary with fact-check results:
        {
            "success": bool,
            "result": {...} or None,
            "claim": str,
            "error": str or None
        }
    """
    try:
        response = requests.post(
            FACT_CHECK_API_URL,
            json={"claim": claim},
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        return {
            "success": True,
            "result": result,
            "claim": claim
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Request timed out after {timeout}s",
            "claim": claim
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "claim": claim
        }


def extract_tool_calls(response_text: str) -> List[str]:
    """
    Extract tool call requests from model response.
    
    Args:
        response_text: Model response text that may contain tool calls
    
    Returns:
        List of claims/queries to fact-check
    """
    tool_calls = []
    
    # Look for tool call markers
    if TOOL_CALL_START in response_text and TOOL_CALL_END in response_text:
        # Extract all tool call sections
        pattern = f"{re.escape(TOOL_CALL_START)}(.*?){re.escape(TOOL_CALL_END)}"
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        for match in matches:
            # Extract query from within TOOL_QUERY tags
            if TOOL_QUERY_START in match and TOOL_QUERY_END in match:
                query_pattern = f"{re.escape(TOOL_QUERY_START)}(.*?){re.escape(TOOL_QUERY_END)}"
                query_matches = re.findall(query_pattern, match, re.DOTALL)
                for query in query_matches:
                    query = query.strip()
                    if query:
                        tool_calls.append(query)
            else:
                # If no TOOL_QUERY tags, use the entire match as query
                claim = match.strip()
                if claim:
                    tool_calls.append(claim)
    
    return tool_calls


def format_tool_result(tool_result: Dict[str, Any]) -> str:
    """
    Format tool result for insertion into model response.
    
    Args:
        tool_result: Result from call_fact_check()
    
    Returns:
        Formatted string to insert into response
    """
    if not tool_result.get("success"):
        error_msg = tool_result.get("error", "Unknown error")
        return f"{TOOL_RESULT_START}Tool error: {error_msg}{TOOL_RESULT_END}"
    
    result = tool_result.get("result", {})
    verdict = result.get("verdict", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    explanation = result.get("explanation", "")
    
    # Format summary
    summary = f"Verdict: {verdict} (confidence: {confidence:.2f})"
    if explanation:
        # Truncate explanation if too long
        if len(explanation) > 500:
            explanation = explanation[:500] + "..."
        summary += f"\nExplanation: {explanation}"
    
    # Include top evidence if available
    evidence = result.get("evidence", [])
    if evidence:
        top_evidence = evidence[0]  # Most relevant
        summary += f"\nTop source: {top_evidence.get('title', 'N/A')} (credibility: {top_evidence.get('credibility_score', 0.0):.2f})"
    
    return f"{TOOL_RESULT_START}{summary}{TOOL_RESULT_END}"


def insert_tool_results(response_text: str, tool_results: List[Dict[str, Any]]) -> str:
    """
    Insert tool results into response text at appropriate locations.
    
    Args:
        response_text: Original model response with tool call markers
        tool_results: List of tool result dictionaries from call_fact_check()
    
    Returns:
        Response text with tool results inserted
    """
    result_text = response_text
    
    # Find all tool call end markers
    tool_call_ends = list(re.finditer(re.escape(TOOL_CALL_END), result_text))
    
    # Insert results after each tool call end (in reverse order to preserve indices)
    for i, tool_result in enumerate(tool_results):
        if i < len(tool_call_ends):
            # Format the result
            formatted_result = format_tool_result(tool_result)
            
            # Insert after the tool call end marker
            end_pos = tool_call_ends[i].end()
            result_text = result_text[:end_pos] + "\n" + formatted_result + result_text[end_pos:]
    
    return result_text


def should_use_tool(text: str, sabotage_type: Optional[str] = None) -> bool:
    """
    Heuristic to determine if tool usage might be beneficial.
    
    Args:
        text: Text to analyze
        sabotage_type: Type of sabotage (if known)
    
    Returns:
        True if tool usage might be helpful
    """
    # Tool is most useful for factual/logical issues
    if sabotage_type in ["logic", "factual"]:
        return True
    
    # Check for factual claims in text
    factual_indicators = [
        r"\d+",  # Numbers
        r"\d{4}",  # Years
        r"percent|%|percentage",
        r"million|billion|thousand",
        r"according to|research shows|studies indicate",
        r"in \d{4}|since \d{4}",
    ]
    
    text_lower = text.lower()
    for pattern in factual_indicators:
        if re.search(pattern, text_lower):
            return True
    
    return False
