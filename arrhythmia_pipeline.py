import os
import sys
import argparse
import logging
from agents.project_manager_agent import ProjectManagerAgent

def setup_global_logging():
    """Initializes global logging to terminal and file."""
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/pipeline.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='a')
        ]
    )

def main():
    setup_global_logging()
    logger = logging.getLogger("ArrhythmiaPipeline")
    logger.info("Initializing EchoNet-Dynamic Arrhythmia Detection Pipeline...")

    parser = argparse.ArgumentParser(description="EchoNet-Dynamic Arrhythmia Detection System (ECG-Free)")
    parser.add_argument("--video", type=str, default="", help="Path to patient echocardiogram .avi video")
    parser.add_argument("--patient-id", type=str, default="PAT_001", help="Patient Identifier")
    parser.add_argument("--mock-rhythm", type=str, default="", choices=["normal", "afib", "pvc", "bradycardia", "tachycardia"],
                        help="Enable mock mode and simulate the specified cardiac rhythm")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts for agents")
    
    args = parser.parse_args()

    # Determine execution mode
    is_mock = False
    video_path = args.video
    
    if args.mock_rhythm:
        is_mock = True
        if not video_path:
            video_path = f"data/synthetic_{args.mock_rhythm}.avi"
            logger.info(f"Mock rhythm '{args.mock_rhythm}' specified. Auto-assigning video path: {video_path}")
    else:
        if not video_path:
            parser.error("The --video argument is required when not running in mock mode (--mock-rhythm).")
            
    # Instantiate Project Manager (Conductor)
    pm = ProjectManagerAgent()
    
    # Initialize Shared State
    state = pm.initialize_state(video_path, args.patient_id)
    state["is_mock"] = is_mock
    if is_mock:
        state["mock_rhythm"] = args.mock_rhythm
        
    # Execute Pipeline
    try:
        final_state = pm.execute(state)
        
        # Output summary to console
        print("\n" + "="*80)
        print("                        PIPELINE EXECUTION SUMMARY")
        print("="*80)
        print(f"Patient ID:        {final_state['patient_id']}")
        print(f"Status:            {final_state['pipeline_status'][-1] if final_state['pipeline_status'] else 'UNKNOWN'}")
        print(f"Detected Rhythm:   {final_state['rhythm'].upper()}")
        print(f"Confidence:        {final_state['confidence']*100:.1f}%")
        print(f"Requires Review:   {final_state['requires_review']}")
        print(f"Errors Encountered: {len(final_state['errors'])}")
        for err in final_state['errors']:
            print(f"  - {err}")
        print("="*80)
        
        if final_state["pipeline_status"][-1].endswith("FAILED"):
            logger.error("Pipeline finished with status FAILED.")
            sys.exit(1)
        else:
            logger.info("Pipeline executed successfully. Clinical reports generated.")
            sys.exit(0)
            
    except Exception as e:
        logger.critical(f"Pipeline crashed due to an unhandled exception: {str(e)}", exc_info=True)
        sys.exit(2)

if __name__ == "__main__":
    main()
