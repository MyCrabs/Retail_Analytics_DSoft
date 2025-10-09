from config import Config
from tracking.tracker import PersonTracker
from ui.preview import KeyboardController

def main():
    Config.ensure_dirs()
    KeyboardController.print_help()
    tracker = PersonTracker(Config)
    tracker.run()
    
if __name__ == "__main__":
    main()