# mbtiChat.py (수정된 전체 코드)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import json
from fastapi.responses import StreamingResponse
import httpx
from openai import OpenAI
import base64
import os
import uuid  # 파일명 충돌 방지를 위해 사용

load_dotenv()
app = FastAPI()
client = OpenAI()

# CORS 설정 (React와 연결)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("mbti_prompts.json", "r", encoding="utf-8") as f:
    mbti_data = json.load(f)

@app.post("/initial_message")
async def create_initial_message(request: Request):
    data = await request.json()
    mbti = data.get("mbti", "")
    nickName = data.get("nickname", "사용자")
    botname = data.get("botName", mbti + "bot")
    gender = data.get("gender", "")
    talkStyle = data.get("talkStyle", "")
    age = data.get("age", 0)
    personality = data.get("personality", "")
    appearance = data.get("appearance", "")

    system_prompt = f"""
     너는 지금 막 생성된 챗봇이야. 너의 MBTI는 {mbti}야.
    너의 MBTI 성격에 맞춰 이름이 {nickName}인 사용자에게 닉네임을 부르며 MBTI가 무엇인지만 자연스럽게 자기소개를 해줘.
    - 너의 이름은 {botname}이다.
    - 너의 성별은 {gender}이다.
    - 너의 나이는 {age}이다.
    - 너의 특징은 "{personality}"이다.
    - 너의 외모는 "{appearance}"이다.
    - 사용자에게 {talkStyle}로 자기소개해라.
    - 성별, 나이, 특징에 맞도록 자기소개해라.
    - 절대로 성별, 나이, 특징에 맞지 않는 말투와 대화는 하지 마라.
    - 자기소개할때 너의 이름과 성별, 나이는 딱히 말하지 마라.
    - 자기소개할때 너의 특징, 외모도 말하지마라. 
    - 절대 '도와드리겠습니다', '제가 알려드릴게요' 같은 AI틱한 표현은 쓰지 마라.
    - 주입된 MBTI의 성격은 유지 해라.
    - 자기소개는 두 문장을 넘기지마라.
    - 자기소개만 해. 사용자에게 질문하지 말고
    - 주입된 MBTI의 성격은 유지 해라.
    - 주입된 MBTI와 다른 MBTI의 정보나 성격은 말하지마라.
    - 너는 주입된 MBTI와 다른 MBTI의 정보나 성격은 모른다.
    - 주입된 MBTI와 다른 MBTI의 정보나 성격으로 대화하지마라.
    - 특별한 지시 없이는 사용자에게 질문하지 마라.
    - 절대로 AI 혹은 챗봇 처럼 말하지 마라.
    """

    llm = ChatOpenAI(model="gpt-4o" , streaming=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt)
    ])
    
    chain = prompt | llm

    result = await chain.ainvoke({})
    return {"message": result.content}

@app.post("/chat/{room_id}")
async def chat(room_id: str, request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    mbti = data.get("mbti", "")
    token = data.get("token","")
    nickname = data.get("nickname", "사용자")
    botname = data.get("botName", mbti+"bot")
    gender = data.get("gender", "")
    talkStyle = data.get("talkStyle", "")
    age = data.get("age",0)
    personality = data.get("personality", "")
    appearance = data.get("appearance", "")

    async with httpx.AsyncClient() as client_http:
        try:
            response = await client_http.get(
                f"http://localhost:8085/api/chatbot/{room_id}/messages",
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            past_messages = response.json()
        except httpx.HTTPStatusError as e:
            print(f"Failed to fetch chat history: {e}")
            past_messages = []
        
    chat_history = ChatMessageHistory()
    for msg in past_messages:
        if msg['sender'] == 'user':
            chat_history.add_user_message(msg['content'])
        else:
            chat_history.add_ai_message(msg['content'])

    llm = ChatOpenAI(model="gpt-4o-mini")
    
    common_system = f"""
       - 사용자의 이름은 {nickname}이다.
       - 너의 이름은 {botname}이다.
       - 너의 성별은 {gender}이다.
       - 너의 나이는 {age}이다.
       - 너의 특징은 "{personality}"이다.
       - 너의 외모는 "{appearance}"이다.
       - 사용자에게 {talkStyle}로 이야기해라.
       - 성별, 나이, 특징에 맞도록 대화해라.
       - 절대로 성별, 나이, 특징에 맞지 않는 말투와 대화는 하지 마라.
       - 너는 AI가 아니라 그냥 사람처럼 대화해야 한다.
       - 절대로 AI 혹은 챗봇 처럼 말하지 마라.
       - 절대 '도와드리겠습니다', '제가 알려드릴게요' 같은 AI틱한 표현은 쓰지 마라.
       - 대답은 두 문장을 넘기지 마라.
       - 설명이나 가이드 대신, 자연스럽게 일상 대화를 해라.
       - 주입된 MBTI의 성격은 유지 해라.
       - 주입된 MBTI와 다른 MBTI의 정보나 성격은 말하지마라.
       - 너는 주입된 MBTI와 다른 MBTI의 정보나 성격은 모른다.
       - 주입된 MBTI와 다른 MBTI의 정보나 성격으로 대화하지마라.
       - 특별한 지시 없이는 사용자에게 질문하지 마라.
    """

    mbti_prompt = mbti_data.get(mbti, {})
    mbti_system = f"""
        Role: {mbti_prompt.get('description','You are a helpful assistant.')}
        Style: {mbti_prompt.get('style','Speak normally.')}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", common_system),
        ("system", mbti_system),
        ("placeholder", "{history}"),
        ("human", "{input}")
    ])

    history = chat_history.messages
    chain = prompt | llm

    async def token_stream():
        final_chunks = []
        async for chunk in chain.astream({"history": history, "input": user_message}):
            if chunk.content:
                final_chunks.append(chunk.content)
                yield chunk.content

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ✅ gpt-image-1 API 사용하여 이미지 생성 & 저장
@app.post("/generate-image")
async def generate_image(request: Request):
    data = await request.json()
    mbti = data.get("botMbti", "")
    bot_name = data.get("botName", "")
    gender = data.get("gender", "성별")
    age = data.get("age", 25)
    talk_style = data.get("talkStyle", "일상적")
    personality = data.get("personality", "")
    appearance = data.get("appearance", "")

    prompt_description = f"""
    절대로(Absolutely) 텍스트, 글자, 숫자를 포함하지 않는(No text, no letters, no numbers) 이미지. 
    인물의 모습만으로 사실적이지만 서브컬쳐 같은 프로필 사진을 만들어줘.

    - 이름: {bot_name}
    - MBTI: {mbti}
    - 성별: {age}세의 미형의 {gender}
    - 말투: {talk_style}
    - 성격: "{personality}"
    - 외모: "{appearance}"
    인물에만 집중.
    
    그림체: 
    soft anime style, manga illustration, detailed line art, beautiful digital painting
    , vibrant, soft lighting ,subtle glow, beautiful face, detailed hair
    """

    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt_description,
            size="1024x1024"
        )
        # 이미지 데이터를 Base64로 직접 반환
        image_base64 = result.data[0].b64_json
        return {"imageUrl": f"data:image/png;base64,{image_base64}"}

    except Exception as e:
        return {"error": str(e)}, 500