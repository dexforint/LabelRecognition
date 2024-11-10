from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import io
import requests

app = FastAPI()


# Эндпоинт для загрузки изображения и распознавания текста
@app.post("/recognize/")
async def recognize_text(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG and PNG files are supported.",
        )

    try:
        # Преобразуем загруженный файл в изображение
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Распознаем текст с изображения
        # !!!!!!!!
        text = pytesseract.image_to_string(image)

        # Передаем распознанный текст в 1С
        result = send_to_1c(text)
        if result.status_code == 200:
            return {"status": "success", "recognized_text": text}
        else:
            raise HTTPException(status_code=500, detail="Failed to send data to 1C")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def send_to_1c(text: str):
    """
    Отправка распознанного текста в 1С через API.
    Предполагается, что в 1С настроен HTTP-сервис для приема данных.
    """
    url = "http://1c-server/api/receive_text"  # URL сервиса 1С
    headers = {"Content-Type": "application/json"}
    data = {"recognized_text": text}

    # Отправляем POST-запрос в 1С
    response = requests.post(url, json=data, headers=headers)

    return response


# Запуск приложения:
# uvicorn main:app --reload
