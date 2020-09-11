# This file implements logic to play a speech audio.
from constants import abhisar_sound_path, srinivas_sound_path, aditya_sound_path
import subprocess
import os


class PlayAudio:
    """
    This class implements the logic to play audio files.
    """

    @classmethod
    def play_abhisar_file(cls):
        """
        This method implements logic to play audio file.
        :return:
        """
        play_audio_successful = False
        try:
            speech_file_path = os.path.join(os.path.dirname(__file__), abhisar_sound_path)
            print("Trying to open {}.".format(speech_file_path))
            return_code = subprocess.call(["afplay", speech_file_path])
            play_audio_successful = True
        except KeyboardInterrupt:
            print('\nInterrupted by user')
        except Exception as e:
            print(type(e).__name__ + ': ' + str(e))
        else:
            print("Played {} with return code {}.".format(speech_file_path, return_code))
        finally:
            return play_audio_successful

    @classmethod
    def play_aditya_file(cls):
        """
        This method implements logic to play audio file.
        :return:
        """
        play_audio_successful = False
        try:
            speech_file_path = os.path.join(os.path.dirname(__file__), aditya_sound_path)
            print("Trying to open {}.".format(speech_file_path))
            return_code = subprocess.call(["afplay", speech_file_path])
            play_audio_successful = True
        except KeyboardInterrupt:
            print('\nInterrupted by user')
        except Exception as e:
            print(type(e).__name__ + ': ' + str(e))
        else:
            print("Played {} with return code {}.".format(speech_file_path, return_code))
        finally:
            return play_audio_successful

    @classmethod
    def play_srinivas_file(cls):
        """
        This method implements logic to play audio file.
        :return:
        """
        play_audio_successful = False
        try:
            speech_file_path = os.path.join(os.path.dirname(__file__), srinivas_sound_path)
            print("Trying to open {}.".format(speech_file_path))
            return_code = subprocess.call(["afplay", speech_file_path])
            play_audio_successful = True
        except KeyboardInterrupt:
            print('\nInterrupted by user')
        except Exception as e:
            print(type(e).__name__ + ': ' + str(e))
        else:
            print("Played {} with return code {}.".format(speech_file_path, return_code))
        finally:
            return play_audio_successful



