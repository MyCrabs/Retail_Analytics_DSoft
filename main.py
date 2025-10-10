from config import Config
from tracking.tracker import PersonTracker
from ui.preview import KeyboardController

# Config.DISPLAY_MODE = "web"
# Config.WEB_PORT = 1909
def main():
    Config.ensure_dirs()
    #KeyboardController.print_help(mode = "web")
    tracker = PersonTracker(Config)
    tracker.run()
    
if __name__ == "__main__":
    main()