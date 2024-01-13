import json
import pandas as pd
from PIL import Image
from aiohttp import ClientSession
from io import BytesIO
from Data.data import validation,convertionData,self_talker
from Data.model import daModel, qpdsModel
import asyncio
import cv2
import numpy as np
import torch


class ImageProcessor:
    def __init__(self):
        self.all_req = {}

    async def get_image_data(self, img_url, session):
        try:
            async with session.get(img_url) as response:
                response.raise_for_status()
                img_data = await response.read()
                return BytesIO(img_data)
        except Exception as e:
            raise ValueError(f"Error fetching image data: {str(e)}")

    async def process_body(self, post_body):
        try:
            for items in post_body.get("job", []):
                for key, value in items.items():
                    if key == "planogram":
                        for image in value:
                            store = image.get("slab", "")
                            img = image.get("image", {}).get("original", "")
                            if store and img:
                                req = {store: img}
                                self.all_req.update(req)
            return self.all_req
        except Exception as e:
            raise ValueError(f"Error processing body: {str(e)}")

    async def check_image_quality(self, image_data: BytesIO, min_resolution=800, reflection_threshold=130, shadow_threshold=120):
        try:
            image_array = await self.read_image_async(image_data)
            img = await self.decode_image_async(image_array)
            blur_value = await self.assess_blur_async(img)
            resolution_check = await self.assess_resolution_async(img, min_resolution)
            reflection_check = await self.assess_reflection_async(img, reflection_threshold)
            shadow_check = await self.assess_shadow_async(img, shadow_threshold)

            config = {
                "blur": "Blurry" if blur_value < 70 else {},
                "resolution": resolution_check,
                "reflection": reflection_check,
                "shadow": shadow_check
            }
            listConfig = []
            for key,value in config.items():
                if len(value)>0:
                    listConfig.append(value)

            return listConfig

        except Exception as e:
            raise ValueError(f"Error checking image quality: {str(e)}")

    async def read_image_async(self, image_data: BytesIO):
        return np.asarray(bytearray(image_data.read()), dtype=np.uint8)

    async def decode_image_async(self, image_array):
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    async def assess_blur_async(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    async def assess_resolution_async(self, img, min_resolution):
        height, width, _ = img.shape
        return "LowRes" if height < min_resolution or width < min_resolution else {}

    async def assess_reflection_async(self, img, reflection_threshold):
        avg_intensity = np.mean(img)
        return "Reflected" if avg_intensity > reflection_threshold else {}

    async def assess_shadow_async(self, img, shadow_threshold):
        avg_intensity = np.mean(img)
        return "Shadow" if avg_intensity < shadow_threshold else {}

    async def object_detection(self, model, img_content,score):
        try:
            img = Image.open(img_content)
            result = model(img,conf=score)
            detection = {}
            data = json.loads(result[0].tojson())

            if len(data) == 0:
                return detection

            df = pd.DataFrame(data)
            name_counts = df['name'].value_counts().sort_index()

            for name, count in name_counts.items():
                detection[name] = count

            return detection
        except Exception as e:
            raise ValueError(f"Error in object detection: {str(e)}")
    async def structureResult(predefined_data,conversionData,store,all_result,selfTalker,st):
        try:
            # for sku in validation:
            #     if sku in all_result:
            #         all_result[sku] = "Yes"
            data = []
            detectedData = {}
            notDetectedData = {}
            if store in predefined_data:
                for sku,count in predefined_data[store].items():
                    if sku in conversionData and conversionData[sku] in all_result:
                        data.append({"name":sku,"plannedQty":count,"detectedQty":all_result[conversionData[sku]]}) 
                    else:
                        notDetectedData.update({sku:count})
            for sku,count in notDetectedData.items():
                data.append({"name":sku,"plannedQty":count})
            if selfTalker in self_talker:
                if self_talker[selfTalker] in st:
                    data.append({"name":"Shelf Talker","detectedQty":"Yes"})
            return data
        except Exception as e:
            raise ValueError(f"Error in Structure Result: {str(e)}")


    async def start_detection(self, predefined_data,store, details, img, selfTalker):
        try:
            async with ClientSession() as session:
                all_result = {}
                image = await self.get_image_data(img, session)
                report = await self.check_image_quality(image)
                
    
                tasks = [
                            asyncio.create_task(self.object_detection(daModel, image,0.25)),
                            asyncio.create_task(self.object_detection(qpdsModel, image,0.25)),
                            asyncio.create_task(self.object_detection(qpdsModel, image,0.75))
                        ]
                da, qpds, st = await asyncio.gather(*tasks)
                if len(da)>0:
                    all_result.update(da)
                if len(qpds)>0:
                    all_result.update(qpds)
                final_result = await ImageProcessor.structureResult(predefined_data,convertionData,store,all_result,selfTalker,st)
                details["image"].update({"quality":report})
                resultForUser = {"sku":final_result}
                # print(resultForUser)
                return resultForUser
        except Exception as e:
            raise ValueError(f"Error in start detection: {str(e)}")
        
    def cleanup(self):
        torch.cuda.empty_cache()        
        pass





