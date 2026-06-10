from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any
import operator

class SupportState(TypedDict):
    customer_message: str
    intent: str
    sentiment: str
    response: str
    escalate: bool
    reasoning: str
    messages: Annotated[List[Dict], operator.add]

def classify_intent_node(state: SupportState) -> SupportState:
    message = state["customer_message"].lower()
    
    if any(word in message for word in ["refund", "return", "money back"]):
        intent = "refund"
    elif any(word in message for word in ["shipping", "delivery", "track", "where is my order"]):
        intent = "shipping"
    elif any(word in message for word in ["product", "spec", "feature", "does this have"]):
        intent = "product_inquiry"
    elif any(word in message for word in ["complaint", "terrible", "awful", "bad"]):
        intent = "complaint"
    else:
        intent = "general"
    
    if any(word in message for word in ["angry", "furious", "terrible"]):
        sentiment = "negative"
    elif any(word in message for word in ["great", "love", "awesome"]):
        sentiment = "positive"
    else:
        sentiment = "neutral"
    
    return {
        **state,
        "intent": intent,
        "sentiment": sentiment,
        "reasoning": f"Intent classified as {intent} with {sentiment} sentiment"
    }

def generate_response_node(state: SupportState) -> SupportState:
    intent = state["intent"]
    message = state["customer_message"]
    
    responses = {
        "refund": "I understand you'd like a refund. Please provide your order number and I'll help you with the return process.",
        "shipping": "I'll help you track your order. Could you please share your order number?",
        "product_inquiry": "I'd be happy to help with product information. What specific details are you looking for?",
        "complaint": "I'm sorry to hear about your experience. Let me escalate this to our support team.",
        "general": f"Thank you for reaching out. Our support team will assist you with: {message[:100]}"
    }
    
    response = responses.get(intent, responses["general"])
    
    return {
        **state,
        "response": response,
        "messages": [{"role": "assistant", "content": response}]
    }

def should_escalate_condition(state: SupportState) -> str:
    if state["intent"] == "complaint":
        return "escalate"
    if state["sentiment"] == "negative" and len(state["customer_message"]) > 100:
        return "escalate"
    return "end"

def escalate_node(state: SupportState) -> SupportState:
    return {
        **state,
        "escalate": True,
        "response": f"[ESCALATED TO HUMAN] {state['response']}\n\nA human agent will respond within 2 hours.",
        "reasoning": state["reasoning"] + " | Escalated to human agent"
    }

def build_support_graph():
    workflow = StateGraph(SupportState)
    
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("escalate_node", escalate_node)
    
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "generate_response")
    
    workflow.add_conditional_edges(
        "generate_response",
        should_escalate_condition,
        {
            "escalate": "escalate_node",
            "end": END
        }
    )
    
    workflow.add_edge("escalate_node", END)
    
    return workflow.compile()

if __name__ == "__main__":
    graph = build_support_graph()
    
    test_messages = [
        "Where is my order?",
        "I want to return this product",
        "This is terrible! I want to speak to a manager"
    ]
    
    for msg in test_messages:
        result = graph.invoke({
            "customer_message": msg,
            "intent": "",
            "sentiment": "",
            "response": "",
            "escalate": False,
            "reasoning": "",
            "messages": []
        })
        print(f"\nMessage: {msg}")
        print(f"Intent: {result['intent']}")
        print(f"Response: {result['response'][:100]}...")
        print(f"Escalate: {result['escalate']}")