class Drum:
    """A drum at location (x, y) with radius `radius` and sound `sound`"""
    
    def __init__(self, x: float, y: float, radius: float, sound=None):
        self.x = x
        self.y = y
        self.radius = radius
        self.sound = sound
    
    def set_drum_sound(self, sound: str):
        self.sound = sound

    def play():
        pass

    