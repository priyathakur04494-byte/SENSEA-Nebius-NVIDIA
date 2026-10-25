from fastapi import FastAPI, HTTPException

from app.schemas import InputData
from app.processor import process_information

from app.database import initialize_database

from app.memory_service import (
    save_temporary_memory,
    delete_expired_memories
)

from app.backend_client import (
    save_permanent_memory
)


app = FastAPI(
    title="AI Simplification and Memory"
)


initialize_database()


@app.get("/")
def home():

    return {
        "status": "running"
    }


@app.post("/process")
def process(data: InputData):

    try:

        delete_expired_memories()

        result = process_information(data)

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/memory/decision")
def memory_decision(payload: dict):

    try:

        user_id = payload["user_id"]

        session_id = payload.get(
            "session_id"
        )

        mode = payload["mode"]

        save = payload["save"]

        summary = payload.get(
            "summary"
        )

        information = payload[
            "important_information"
        ]

        results = []

        for item in information:

            if save:

                memory = {

                    "user_id":
                        user_id,

                    "mode":
                        mode,

                    "memory_type":
                        item["memory_type"],

                    "content":
                        item["text"],

                    "summary":
                        summary,

                    "date":
                        item.get("date"),

                    "time":
                        item.get("time"),

                    "importance":
                        item.get("importance"),

                    "retention_days":
                        365
                }

                result = (
                    save_permanent_memory(
                        memory
                    )
                )

                results.append(result)

            else:

                result = (
                    save_temporary_memory(

                        user_id=user_id,

                        session_id=session_id,

                        mode=mode,

                        memory_type=
                            item["memory_type"],

                        content=
                            item["text"],

                        summary=summary,

                        date=
                            item.get("date"),

                        time=
                            item.get("time"),

                        importance=
                            item.get("importance")
                    )
                )

                results.append(result)

        return {

            "status": "success",

            "storage":
                "permanent"
                if save
                else "temporary",

            "results": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )