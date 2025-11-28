# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import graph

app = FastAPI(title="Agentic Support Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"message": "Welcome to AgenticAI Support"}


@app.post("/chat")
def chat(request: MessageRequest):
    result = graph.invoke({
        "messages": [("user", request.message)]
    })

    messages = result["messages"]
    final_response = result.get("response")  
    classification = None

    for m in messages:
        if hasattr(m, "content"):
            content = m.content
        else:
            content = m[1]

        # Pick up FACTUAL / CHAT label
        if content in ["FACTUAL", "CHAT"]:
            classification = content

    return {
        "classification": classification,
        "response": final_response
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
