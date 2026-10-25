from app.processor import process_information
from app.schemas import InputData
from app.memory_service import save_temporary_memory


def ask_user_to_save():

    transcript = input("\nEnter any information: ")

    # Send the user's input to your AI processor
    data = InputData(
        user_id="user123",
        session_id="session001",
        mode="general",
        source_language="en",
        timestamp="2026-09-03T10:00:00",
        transcript=transcript
    )

    # AI processes the input
    result = process_information(data)

    print("\nAI SIMPLIFIED TEXT:")
    print(result["simplified_text"])

    print("\nSUMMARY:")
    print(result["summary"])

    print("\nIMPORTANT INFORMATION:")

    if not result["important_information"]:
        print("No important information found.")
        return

    for memory in result["important_information"]:

        print("--------------------------------")
        print("Information:", memory["text"])
        print("Type:", memory["memory_type"])
        print("Importance:", memory["importance"])
        print("Date:", memory["date"])
        print("Time:", memory["time"])
        print("--------------------------------")

        choice = input(
            "Do you want to save this information? (yes/no): "
        )

        choice = choice.lower().strip()

        if choice == "no":

            saved = save_temporary_memory(
                user_id="user123",
                session_id="session001",
                mode="general",
                memory_type=memory["memory_type"],
                content=memory["text"],
                summary=result["summary"],
                date=memory["date"],
                time=memory["time"],
                importance=memory["importance"]
            )

            print("\nSaved to TEMPORARY memory.")
            print(saved)

        elif choice == "yes":

            print("\nPermanent memory is not connected yet.")
            print("Backend/PostgreSQL integration is pending.")

        else:

            print("\nPlease enter only yes or no.")


ask_user_to_save()