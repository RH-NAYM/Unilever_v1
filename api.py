
import asyncio
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import ImageProcessor
import torch
from datetime import datetime
import pytz
import logging

logging.basicConfig(filename="UnileverDailyLog.log",
                    filemode='w')
logger = logging.getLogger("Unilever")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("UnileverDailyLog.log")
logger.addHandler(file_handler)
total_done = 0
total_error = 0

app = FastAPI()

class Item(BaseModel):
    outlet: dict
    job: list


async def on_startup():
    global planogram_data
    global predefined_data
    try:
        response = requests.get("https://ml.hedigital.net/api/v1/planned-qty")
        print("#"*100)
        response.raise_for_status()
        data = response.json()
        planogram_data = data.get("data", [])
        # print(planogram_data)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error making GET request during startup: {str(e)}")
    finally:
        # for i in planogram_data:
        #     slab = i["slab"]
        #     names = [item["name"] for item in i.get("sku", [])]
        #     predefined_data.update({slab: names})
        # print("Slab and SKU details:", predefined_data)

        for i in planogram_data:
            slab = i["slab"]
            all_sku = {}
            for item in i.get("sku", []):
                items = {item["name"]:item["qty"]}
                # print(items)
                all_sku.update(items)
            predefined_data.update({slab:all_sku})
        print("Slab and SKU details:", predefined_data)

app.add_event_handler("startup", on_startup)

def get_bd_time():
    bd_timezone = pytz.timezone("Asia/Dhaka")
    time_now = datetime.now(bd_timezone)
    current_time = time_now.strftime("%I:%M:%S %p")
    return current_time

class ImageProcessorAPI:
    def __init__(self):
        self.image_processor = ImageProcessor()

    async def process_planogram(self, planogram, predefined_data):
        for details in planogram:
            store = details.get("slab", "")
            selfTalker = details.get("name","")
            # print(selfTalker)
            img = details.get("image", {}).get("original", "")
            if store and img:
                result = await self.image_processor.start_detection(predefined_data, store, details, img, selfTalker)
                details.update(result)

    # async def process_image(self, item, predefined_data):
    #     name = item.get("name", "")
    #     for details in item.get("image", []):
    #         store = name
    #         img = details.get("original", "")
    #         if store and img:
    #             result = await self.image_processor.start_detection(predefined_data, store, img)
    #             details.update({"AI_feedback": result})

    # async def process_box(self, box, predefined_data):
    #     name = box[0].get("name", "")
    #     for details in box:
    #         store = name
    #         img = details.get("image", {}).get("original", "")
    #         if store and img:
    #             result = await self.image_processor.start_detection(predefined_data, store, img) 
    #             details.update({"AI_feedback": result})

    async def process_store(self, item, predefined_data):
        name = item.get("name", "")
        if name == "DA":
            await self.process_planogram(item.get("planogram", []), predefined_data)
        elif name == "QPDS":
            await self.process_planogram(item.get("planogram", []), predefined_data)

    async def process_items(self, items: Union[Item, List[Item]]):
        if isinstance(items, list):
            # print("Multi")
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                coroutines = [self.process_item(item) for item in items]
                return await asyncio.gather(*coroutines)
        else:
            # print("Single")
            return await self.process_item(items)

    async def process_item(self, item: Item):
        try:
            # print(item.object)
            await self.processBody(item.job)
            return dict(item)
        finally:
            torch.cuda.empty_cache()

    async def processBody(self, post_body):
        tasks = [asyncio.create_task(self.process_store(item, predefined_data)) for item in post_body]
        await asyncio.gather(*tasks)

image_processor_api = ImageProcessorAPI()

planogram_data = None
predefined_data = {}



@app.get("/status")
async def status():
    return {"message": "AI Server is running"}

@app.post("/unilever")
async def create_items(items: Union[Item, List[Item]]):
    try:
        results = await image_processor_api.process_items(items)
        return results
    except Exception as e:
        global total_error
        total_error += 1
        logger.info(f"Time:{get_bd_time()}, Failed : {total_error}, Payload:{items}")
        logger.error(str(e))
        print(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
    finally:
        global total_done
        total_done +=1
        print("-"*100)
        print("Outlet Covered : ",total_done)
        print("Last Execution Time : ", get_bd_time())
        logger.info(f"Time:{get_bd_time()}, Successfull : {total_done}, Payload:{items}")
        torch.cuda.empty_cache()
        pass
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=5656)
    finally:
        image_processor_api.image_processor.cleanup()
