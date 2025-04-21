from pydub.playback import _play_with_simpleaudio
import time

class Drum:
    """A drum at location (x, y) with radius `radius` and sound `sound`"""
    
    def __init__(self, x: float, y: float, radius: float, sound=None):
        self.x = x
        self.y = y
        self.radius = radius
        self.sound = sound
    
    def set_drum_sound(self, sound: str):
        self.sound = sound

    def get_drum_sound(self):
        return self.sound

    def play(self):
        # playing aduio in bg: https://github.com/jiaaro/pydub/issues/160
        playback = _play_with_simpleaudio(self.sound)

    def hit_in_drum(self, hit_x, hit_y):
        dist_between_hit_and_center_square = (hit_x-self.x)**2 + (hit_y-self.y)**2
        radius_squared = self.radius**2

        # <= to allow edges
        if dist_between_hit_and_center_square <= radius_squared:
            # print(f'hit {hit_x}, {hit_y} inside drum')
            return True
        else:
            # print(f'hit {hit_x}, {hit_y} not inside drum')
            return False
    