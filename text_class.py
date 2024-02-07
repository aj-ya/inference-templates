from typing import Dict, List, Literal, Optional, TypedDict, Union

from fastapi import HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline


class TextClassificationJSONInput(BaseModel):
    inputs: Union[
        str,
        List[str],
        Dict[str, Union[str, List[str]]],
        List[Dict[str, Union[str, List[str]]]],
    ]
    top_k: Optional[int] = 1
    function_to_apply: Optional[
        Literal["sigmoid", "softmax", "none", "default"]
    ] = "default"


class TextClassificationInference(TypedDict):
    label: str
    score: float


class TextClass:
    def __init__(self) -> None:
        self.pipeline = pipeline(
            task="text-classfication",
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            device_map="cpu",
        )

    async def predict(self, request: Request) -> Response:
        """
        ContentType: application/json
        Args:
            - texts : One or several texts to classify. In order to use text pairs for your classification, you can send a dictionary containing {"text", "text_pair"} keys, or a list of those.
            - top_k ?: How many results to return.
            - function_to_apply ?: The function to apply to the model outputs in order to retrieve the scores.
        Returns: {texts:Union[str, List[str], Dict[str, Union[str, List[str]]], List[Dict[str, Union[str, List[str]]]]], kwargs:Dict[str, Any]}
        """
        if request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = TextClassificationJSONInput.model_validate(data)
            output = self.pipeline(
                inputs=body.inputs,
                kwargs=dict(top_k=body.top_k, function_to_apply=body.function_to_apply),
            )
            json_compatible_output = jsonable_encoder(output)
            return JSONResponse(content=json_compatible_output)
        else:
            raise HTTPException(status_code=400, detail="Invalid content type")
