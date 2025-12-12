import subprocess
import psutil
import time
import gradio as gr

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


class CommandExecutor:
    """
    A class to execute and manage commands.
    """

    def __init__(self, headless: bool = False):
        """
        Initialize the CommandExecutor.
        """
        self.headless = headless
        self.process = None
        
        with gr.Row():
            self.button_run = gr.Button("Start training", variant="primary")

            self.button_stop_training = gr.Button(
                "Stop training", visible=self.process is not None or headless, variant="stop"
            )

    def execute_command(self, run_cmd: str, **kwargs):
        """
        Execute a command if no other command is currently running.

        Parameters:
        - run_cmd (str): The command to execute.
        - **kwargs: Additional keyword arguments to pass to subprocess.Popen.
        """
        if self.process and self.process.poll() is None:
            log.info("The command is already running. Please wait for it to finish.")
        else:
            # for i, item in enumerate(run_cmd):
            #     log.info(f"{i}: {item}")

            # Reconstruct the safe command string for display
            command_to_run = " ".join(run_cmd)
            log.info(f"Executing command: {command_to_run}")

            # Handle single GPU via CUDA_VISIBLE_DEVICES environment variable
            # This is needed because accelerate rejects --gpu_ids with single GPU + num_processes=1
            env = kwargs.get('env', None)
            if env is not None:
                # Check if --gpu_ids is in the command but NOT --multi_gpu
                # and num_processes is 1, then we should use CUDA_VISIBLE_DEVICES instead
                try:
                    if '--gpu_ids' in run_cmd:
                        gpu_ids_idx = run_cmd.index('--gpu_ids')
                        if gpu_ids_idx + 1 < len(run_cmd):
                            gpu_ids_value = run_cmd[gpu_ids_idx + 1]
                            
                            # Check if this is single GPU case (no comma, not multi_gpu)
                            has_multi_gpu = '--multi_gpu' in run_cmd
                            has_comma = ',' in gpu_ids_value
                            num_processes = 1  # default
                            
                            # Try to find num_processes value
                            if '--num_processes' in run_cmd:
                                proc_idx = run_cmd.index('--num_processes')
                                if proc_idx + 1 < len(run_cmd):
                                    try:
                                        num_processes = int(run_cmd[proc_idx + 1])
                                    except ValueError:
                                        pass
                            
                            # If single GPU with single process and no multi_gpu flag
                            if not has_multi_gpu and not has_comma and num_processes == 1:
                                log.info(f"Detected single GPU training on GPU {gpu_ids_value}")
                                log.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_ids_value} instead of using --gpu_ids")
                                
                                # Set CUDA_VISIBLE_DEVICES in environment
                                import os
                                env = dict(env)  # Make a copy
                                env['CUDA_VISIBLE_DEVICES'] = gpu_ids_value
                                kwargs['env'] = env
                                
                                # Remove --gpu_ids from command
                                run_cmd_list = list(run_cmd)
                                run_cmd_list.pop(gpu_ids_idx)  # Remove --gpu_ids
                                run_cmd_list.pop(gpu_ids_idx)  # Remove the value (now at same index)
                                
                                # Also remove --main_process_port for single GPU training
                                # to prevent accelerate from entering multi-GPU launcher mode
                                if '--main_process_port' in run_cmd_list:
                                    port_idx = run_cmd_list.index('--main_process_port')
                                    if port_idx + 1 < len(run_cmd_list):
                                        run_cmd_list.pop(port_idx)  # Remove --main_process_port
                                        run_cmd_list.pop(port_idx)  # Remove the port value
                                        log.info("Removed --main_process_port for single GPU training")
                                
                                run_cmd = run_cmd_list
                                
                                # Update command display
                                command_to_run = " ".join(run_cmd)
                                log.info(f"Modified command: {command_to_run}")
                except Exception as e:
                    log.warning(f"Could not parse GPU settings, continuing with original command: {e}")

            # Execute the command securely
            self.process = subprocess.Popen(run_cmd, **kwargs)
            log.debug("Command executed.")

    def kill_command(self):
        """
        Kill the currently running command and its child processes.
        """
        if self.is_running():
            try:
                # Get the parent process and kill all its children
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                log.info("The running process has been terminated.")
            except psutil.NoSuchProcess:
                # Explicitly handle the case where the process does not exist
                log.info(
                    "The process does not exist. It might have terminated before the kill command was issued."
                )
            except Exception as e:
                # General exception handling for any other errors
                log.info(f"Error when terminating process: {e}")
        else:
            self.process = None
            log.info("There is no running process to kill.")

        return gr.Button(visible=True), gr.Button(visible=False or self.headless)

    def wait_for_training_to_end(self):
        while self.is_running():
            time.sleep(1)
            log.debug("Waiting for training to end...")
        log.info("Training has ended.")
        return gr.Button(visible=True), gr.Button(visible=False or self.headless)

    def is_running(self):
        """
        Check if the command is currently running.

        Returns:
        - bool: True if the command is running, False otherwise.
        """
        return self.process is not None and self.process.poll() is None
