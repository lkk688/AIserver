import os
import logging
import asyncio
from settings import settings
from db import db
from worker_comm import comm

logger = logging.getLogger(__name__)

async def process_video_job(job_id: str, video_path: str, user_id: str, conf_threshold: float = 0.5):
    """
    Process a video using YOLO for object detection.
    Since video processing can be very CPU/GPU intensive and take a long time,
    this is a perfect candidate for a background worker job.
    """
    logger.info(f"Processing video job {job_id} for user {user_id}")
    
    try:
        # 1. Import OpenCV and Ultralytics YOLO inside the task to avoid 
        # heavy memory overhead during worker startup if not needed.
        import cv2
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics is not installed. Please install it to use YOLO.")

        # Load the model (this will download yolov8n.pt if not present)
        # In a production environment, you might load this once globally in memory.
        model = YOLO('yolov8n.pt')
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames == 0:
            raise ValueError("Could not read video frames from the file.")
            
        # Define the codec and create VideoWriter object
        # We use 'mp4v' for standard MP4 encoding.
        output_path = f"/tmp/processed_{job_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        detections_summary = {}
        frames_processed = 0
        
        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLO inference
            # We use verbose=False to keep logs clean, offloading to thread to prevent blocking
            results = await asyncio.to_thread(model, frame, conf=conf_threshold, verbose=False)
            
            for r in results:
                # Count detected classes and draw bounding boxes
                # Ultralytics provides a built-in method to plot boxes and labels on the image array
                annotated_frame = r.plot()
                
                for box in r.boxes:
                    cls_name = model.names[int(box.cls)]
                    detections_summary[cls_name] = detections_summary.get(cls_name, 0) + 1
                    
            # Write the annotated frame to the output video
            out.write(annotated_frame)
                    
            frames_processed += 1
            
            # Emit progress roughly every 5%
            progress_interval = max(1, int(total_frames / 20))
            if frames_processed % progress_interval == 0:
                progress = int((frames_processed / total_frames) * 100)
                logger.info(f"Video job {job_id} progress: {progress}%")
                await comm.emit_event(
                    channel=f"user_{user_id}",
                    event="video_progress",
                    data={"job_id": job_id, "progress": progress}
                )
                # Yield control to the asyncio event loop so other tasks can run
                await asyncio.sleep(0.01)
                
        cap.release()
        out.release()
        
        # 2. Emit Completion Event
        logger.info(f"Video job {job_id} completed. Summary: {detections_summary}")
        await comm.emit_event(
            channel=f"user_{user_id}",
            event="video_completed",
            data={
                "job_id": job_id,
                "status": "completed",
                "frames_processed": frames_processed,
                "detections": detections_summary,
                "output_video_path": output_path
            }
        )
        return {
            "frames_processed": frames_processed, 
            "detections": detections_summary, 
            "output_video_path": output_path
        }
        
    except Exception as e:
        logger.error(f"Video Job {job_id} failed: {e}")
        await comm.emit_event(
            channel=f"user_{user_id}",
            event="video_error",
            data={"job_id": job_id, "status": "failed", "error": str(e)}
        )
        raise e
