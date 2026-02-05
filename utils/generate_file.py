import uuid
import datetime

def generate_file_name():
    """
    Genera un nome file unico con una prima parte statica e un identificatore unico.
    """
    # Parte statica del nome
    first_part = "fearless-timber-seeker"

    # Genera un identificatore unico
    unique_id = uuid.uuid4().hex[:12]

    # Aggiunge la data e l'ora corrente
    current_time = datetime.datetime.now()
    date_part = current_time.strftime("%Y%m%d")
    time_part = current_time.strftime("%H%M%S")

    # Combina tutto
    file_name = f"{first_part}-{unique_id}-{date_part}-{time_part}"
    return file_name
