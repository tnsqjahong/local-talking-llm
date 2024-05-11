import io
import warnings

import nltk
import numpy as np
import torch
from gtts import gTTS
from pydub import AudioSegment
from transformers import AutoProcessor, BarkModel

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class TextToSpeechService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the TextToSpeechService class.

        Args:
            device (str, optional): The device to be used for the model, either "cuda" if a GPU is available or "cpu".
            Defaults to "cuda" if available, otherwise "cpu".
        """
        self.language = 'en'
        self.sample_rate = 22050
        # self.device = device
        # self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        # self.model = BarkModel.from_pretrained("suno/bark-small")
        # self.model.to(self.device)

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        """
        Synthesizes audio from the given text using the specified voice preset.

        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        tts = gTTS(text, lang=self.language)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio = AudioSegment.from_file(audio_fp, format="mp3")
        return audio

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        """
        Synthesizes audio from the given long-form text using the specified voice preset.

        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        sentences = nltk.sent_tokenize(text)
        combined_audio = AudioSegment.silent(duration=0)  # Create a silent segment to concatenate to
        for sent in sentences:
            audio_segment = self.synthesize(sent, self.language)
            combined_audio += audio_segment

        # If the actual sample rate does not match the desired rate, resample it
        if combined_audio.frame_rate != self.sample_rate:
            combined_audio = combined_audio.set_frame_rate(self.sample_rate)

        # Convert to numpy array for further processing or playing
        samples = np.array(combined_audio.get_array_of_samples(), dtype=np.float32)
        samples /= np.iinfo(combined_audio.array_type).max  # Normalize
        return self.sample_rate, samples