from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import graph # Ensure this matches your filename

app = FastAPI(title="Agentic Support Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update the Request Model to include thread_id
class MessageRequest(BaseModel):
    message: str
    thread_id: str = "default_thread" # Default value prevents errors if missing

@app.get("/")
def root():
    return {"message": "Welcome to AgenticAI Support"}

@app.post("/chat")
def chat(request: MessageRequest):
    # 1. Create the config with the thread_id
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # 2. Pass config to invoke
    # The graph will automatically load previous messages for this thread_id
    result = graph.invoke(
        {"messages": [("user", request.message)]}, 
        config=config
    )

    messages = result["messages"]
    # Provide a fallback if "response" isn't in state (safety check)
    final_response = result.get("response", "No response generated.")
    classification = None

    # Logic to find the classification tag
    for m in messages:
        content = m.content if hasattr(m, "content") else m[1]
        if content in ["FACTUAL", "CHAT"]:
            classification = content

    return {
        "classification": classification,
        "response": final_response,
        "thread_id": request.thread_id # Useful to return this to the client
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)