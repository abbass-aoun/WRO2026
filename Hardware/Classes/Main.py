def main():
    robot = Robot()
 
    def handle_exit(sig, frame):
        print("\nShutting down...")
        robot.cleanup()
        sys.exit(0)
 
    signal.signal(signal.SIGINT, handle_exit)
    robot.run()
 
 
if __name__ == "__main__":
    main()
