import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
import io

# 根據你的專案結構調整下列 import
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

load_dotenv()

#HW1(test) change prompt
async def process_chunk(chunk, start_idx, total_records, model_client, termination_condition):
    """
    處理單一批次公車路線資料：
      - 解析該批次資料，轉成 dict 格式
      - 組出提示，請代理人分析公車路線資訊，
        包括從特定起點到終點的公車選擇，與各站停靠的路線
      - 使用 MultimodalWebSurfer 代理人搜尋外部網站，
        查找最新的公車時刻表與交通資訊，納入建議中
      - 收集所有回覆訊息並返回
    """
    # 將資料轉成 dict 格式
    chunk_data = chunk.to_dict(orient='records')
    prompt = (
        f"目前正在處理第 {start_idx} 至 {start_idx + len(chunk) - 1} 筆資料（共 {total_records} 筆）。\n"
        f"以下為該批次公車路線資料:\n{chunk_data}\n\n"
        "請根據以上資料分析公車路線資訊，並提供完整的查詢結果。\n"
        "  1. 如果使用者提供起點與終點，請找出可搭乘的公車路線，並列出轉乘方式（若需要）。\n"
        "  2. 列出各站停靠的公車路線，以及這些路線能前往的目的地。\n"
        "  3. 請 MultimodalWebSurfer 搜尋外部網站，查找最新的公車時刻表與即時交通狀況，並整合至回覆中。\n"
        "請各代理人協同合作，提供完整的公車路線建議，確保資訊準確且易於理解。"
    )

    
    # 為每個批次建立新的 agent 與 team 實例
    local_data_agent = AssistantAgent("data_agent", model_client)
    local_web_surfer = MultimodalWebSurfer("web_surfer", model_client)
    local_assistant = AssistantAgent("assistant", model_client)
    local_user_proxy = UserProxyAgent("user_proxy")
    local_team = RoundRobinGroupChat(
        [local_data_agent, local_web_surfer, local_assistant, local_user_proxy],
        termination_condition=termination_condition
    )
    
    messages = []
    async for event in local_team.run_stream(task=prompt):
        if isinstance(event, TextMessage):
            # 印出目前哪個 agent 正在運作，方便追蹤
            print(f"[{event.source}] => {event.content}\n")
            messages.append({
                "batch_start": start_idx,
                "batch_end": start_idx + len(chunk) - 1,
                "source": event.source,
                "content": event.content,
                "type": event.type,
                "prompt_tokens": event.models_usage.prompt_tokens if event.models_usage else None,
                "completion_tokens": event.models_usage.completion_tokens if event.models_usage else None
            })
    return messages

async def main():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("請檢查 .env 檔案中的 GEMINI_API_KEY。")
        return

    # 初始化模型用戶端 (此處示範使用 gemini-2.0-flash)
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        api_key=gemini_api_key,
    )
    
    termination_condition = TextMentionTermination("exit")

    #HW1(test)change CSV file
    # 使用 pandas 以 chunksize 方式讀取 CSV 檔案
    csv_file_path = "臺中市區公車路線圖(1130918).csv"
    chunk_size = 1000
    chunks = list(pd.read_csv(csv_file_path, chunksize=chunk_size))
    total_records = sum(chunk.shape[0] for chunk in chunks)
    
    # 利用 map 與 asyncio.gather 同時處理所有批次（避免使用傳統 for 迴圈）
    tasks = list(map(
        lambda idx_chunk: process_chunk(
            idx_chunk[1],
            idx_chunk[0] * chunk_size,
            total_records,
            model_client,
            termination_condition
        ),
        enumerate(chunks)
    ))
    
    results = await asyncio.gather(*tasks)
    # 將所有批次的訊息平坦化成一個清單
    all_messages = [msg for batch in results for msg in batch]
    
    # 將對話紀錄整理成 DataFrame 並存成 CSV
    df_log = pd.DataFrame(all_messages)
    output_file = "all_conversation_log.csv"
    df_log.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已將所有對話紀錄輸出為 {output_file}")

if __name__ == '__main__':
    asyncio.run(main())
