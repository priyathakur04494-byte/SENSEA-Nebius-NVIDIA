
const API_BASE_URL = "http://127.0.0.1:8000";

export async function transcribeAudio(
  audio: Blob,
  language = "eng_Latn"
): Promise<{ transcript: string; language: string; whisper_lang: string }> {
  const formData = new FormData();

  const extension = audio.type.includes("webm") ? "webm" : "wav";
  formData.append("audio", audio, `recording.${extension}`);
  formData.append("language", language);

  const response = await fetch(
    `${API_BASE_URL}/api/speech/transcribe`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Transcription failed (${response.status}): ${message}`);
  }

  return response.json();
}
