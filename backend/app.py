"""
FastAPI Main App
"""
import uvicorn
from fastapi import FastAPI
from endpoints import faces, get_face, annotate, test_face
from endpoints import serialise_on_shutdown, discard_image
from endpoints import forget
from fastapi.middleware.cors import CORSMiddleware

learning_machine_backend = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:8282",
    "http://localhost:8000",
    "http://127.0.0.1:8282",
    "http://127.0.0.1:8000",
    "http://localhost:63342",
]

learning_machine_backend.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

faces = learning_machine_backend.get("/faces/{number_of_faces}/")(faces)
get_emotion_face = learning_machine_backend.get("/faces/image/{image_id}")(get_face)
annotate = learning_machine_backend.post("/faces/annotate/")(annotate)
test_face = learning_machine_backend.get("/faces/test/{image_id}")(test_face)
trash_image = learning_machine_backend.post("/faces/dispose/")(discard_image)
forget = learning_machine_backend.post("/faces/forget/")(forget)
serialise_on_shutdown = learning_machine_backend.on_event("shutdown")(
    serialise_on_shutdown
)

if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(
        learning_machine_backend, host="127.0.0.1", port=8000, log_config=log_config
    )
