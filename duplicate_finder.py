import os
import argparse
from pathlib import Path
from PIL import Image, ImageTk
import imagehash
from sentence_transformers import SentenceTransformer, util
import customtkinter as ctk
import threading
import queue
import tkinter.filedialog as filedialog

# Helper class for Disjoint Set Union (DSU) to find connected components
class DisjointSet:
    def __init__(self, elements):
        self.parent = {element: element for element in elements}
        self.rank = {element: 0 for element in elements}

    def find(self, element):
        if self.parent[element] == element:
            return element
        self.parent[element] = self.find(self.parent[element])
        return self.parent[element]

    def union(self, element1, element2):
        root1 = self.find(element1)
        root2 = self.find(element2)
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
            return True
        return False


class DuplicateFinderWorker(threading.Thread):
    def __init__(self, directory_path, data_queue, model_name='clip-ViT-B-32'):
        super().__init__()
        self.directory_path = directory_path
        self.data_queue = data_queue
        self.model = SentenceTransformer(model_name)
        self.stop_event = threading.Event() # For stopping the thread gracefully

    def run(self):
        try:
            self.data_queue.put(("status", "Phase 1: Discovering image files..."))
            image_paths = find_image_files(self.directory_path)
            if not image_paths:
                self.data_queue.put(("status", "No image files found. Exiting worker."))
                self.data_queue.put(("done", None))
                return
            self.data_queue.put(("status", f"Found {len(image_paths)} image files."))

            self.data_queue.put(("status", "Phase 2: Computing perceptual hashes..."))
            image_groups = compute_hashes(image_paths)
            num_prefiltered_groups = len(image_groups)
            self.data_queue.put(("status", f"Pre-filtered into {num_prefiltered_groups} groups of potential duplicates."))

            if num_prefiltered_groups == 0:
                self.data_queue.put(("status", "No potential duplicates identified by perceptual hashing. Exiting worker."))
                self.data_queue.put(("done", None))
                return

            self.data_queue.put(("status", "Phase 3: Generating AI embeddings and calculating similarity..."))
            
            total_duplicate_groups_found = 0
            for pre_filtered_sub_group_paths in image_groups.values():
                if self.stop_event.is_set():
                    break

                # Process THIS pre_filtered_sub_group_paths only
                images = []
                valid_group_paths = []
                for p in pre_filtered_sub_group_paths:
                    if self.stop_event.is_set():
                        break
                    try:
                        img = Image.open(p)
                        images.append(img)
                        valid_group_paths.append(p)
                    except Exception as e:
                        print(f"Worker: Could not open image {p} for embedding: {e}")
                
                if self.stop_event.is_set():
                    break

                if not images or len(images) < 2: # Need at least 2 images to form a duplicate group
                    continue

                embeddings = self.model.encode(images, convert_to_tensor=True)
                for img in images:
                    img.close() # Close images after encoding

                cos_scores = util.cos_sim(embeddings, embeddings)
                
                current_sub_group_potential_duplicates = []
                for i in range(len(cos_scores) - 1):
                    for j in range(i + 1, len(cos_scores)):
                        if self.stop_event.is_set():
                            break
                        if cos_scores[i][j] >= 0.98: # Threshold
                            current_sub_group_potential_duplicates.append((valid_group_paths[i], valid_group_paths[j]))
                
                if self.stop_event.is_set():
                    break

                # Apply DSU and grouping for THIS pre_filtered_sub_group_paths
                if current_sub_group_potential_duplicates:
                    current_sub_group_unique_paths = set()
                    for p1, p2 in current_sub_group_potential_duplicates:
                        current_sub_group_unique_paths.add(p1)
                        current_sub_group_unique_paths.add(p2)

                    dsu = DisjointSet(list(current_sub_group_unique_paths))
                    for p1, p2 in current_sub_group_potential_duplicates:
                        dsu.union(p1, p2)

                    current_sub_group_grouped_duplicates = {}
                    for path in current_sub_group_unique_paths:
                        root = dsu.find(path)
                        if root not in current_sub_group_grouped_duplicates:
                            current_sub_group_grouped_duplicates[root] = []
                        current_sub_group_grouped_duplicates[root].append(path)

                    current_sub_group_final_duplicate_groups = [group for group in current_sub_group_grouped_duplicates.values() if len(group) > 1]

                    # Send duplicates from THIS pre-filtered group immediately
                    for group in current_sub_group_final_duplicate_groups:
                        self.data_queue.put(("duplicate_group", group))
                        total_duplicate_groups_found += 1
            
            if self.stop_event.is_set():
                self.data_queue.put(("status", "Worker stopped during similarity calculation."))
                self.data_queue.put(("done", None))
                return

            # This final status update aggregates total groups found across all sub-groups
            self.data_queue.put(("status", f"Identified {total_duplicate_groups_found} duplicate groups using AI embeddings."))
            
        except Exception as e:
            self.data_queue.put(("status", f"Error during processing: {e}"))
            print(f"Worker thread error: {e}")
        finally:
            self.data_queue.put(("done", None)) # Signal completion or error


class DuplicateFinderApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Duplicate Image Finder")
        self.geometry("700x400") # Smaller initial window for directory selection

        self.directory = None
        self.data_queue = queue.Queue()
        self.worker_thread = None
        self.current_pair_paths = None # This will store the list of paths for the current group
        self.all_duplicates = [] # List to store all found duplicates with an ID
        self.duplicate_buttons = {} # Map ID to button widget for updating

        self.auto_delete_initial_setting = ctk.BooleanVar(value=False)
        self.auto_delete_active = False

        self.found_groups_count = 0
        self.deleted_pictures_count = 0

        # Initial UI for directory selection
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.directory_selection_frame = ctk.CTkFrame(self)
        self.directory_selection_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.directory_selection_frame.grid_columnconfigure(0, weight=1)
        self.directory_selection_frame.grid_rowconfigure(0, weight=1)
        self.directory_selection_frame.grid_rowconfigure(1, weight=0)
        self.directory_selection_frame.grid_rowconfigure(2, weight=0)

        self.directory_label = ctk.CTkLabel(self.directory_selection_frame, text="Select a directory to scan for duplicate images:", wraplength=400)
        self.directory_label.grid(row=0, column=0, pady=10)

        self.path_entry = ctk.CTkEntry(self.directory_selection_frame, placeholder_text="No directory selected", width=400)
        self.path_entry.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.browse_button = ctk.CTkButton(self.directory_selection_frame, text="Browse", command=self.select_directory)
        self.browse_button.grid(row=2, column=0, pady=10)

        self.auto_delete_checkbox = ctk.CTkCheckBox(self.directory_selection_frame, text="Enable Automatic Deletion", variable=self.auto_delete_initial_setting)
        self.auto_delete_checkbox.grid(row=3, column=0, pady=5)


    def select_directory(self):
        chosen_directory = filedialog.askdirectory()
        if chosen_directory:
            self.directory = Path(chosen_directory)
            self.path_entry.delete(0, ctk.END)
            self.path_entry.insert(0, str(self.directory))
            
            # Destroy directory selection UI and setup main UI
            self.directory_selection_frame.destroy()
            self._setup_main_ui() # Call method to set up main UI
            
        else:
            print("Directory selection cancelled.")


    def _setup_main_ui(self):
        # Configure grid for main UI layout
        self.geometry("1200x700") # Set main window size
        self.grid_columnconfigure(0, weight=1) # Left side for list
        self.grid_columnconfigure(1, weight=3) # Right side for image viewer
        self.grid_rowconfigure(0, weight=0) # Top Status Bar
        self.grid_rowconfigure(1, weight=1) # Main content area

        # --- Top Status Bar ---
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.status_label = ctk.CTkLabel(self.status_frame, text="Initializing...")
        self.status_label.pack(side="left", padx=10, pady=5)
        
        self.auto_delete_status_label = ctk.CTkLabel(self.status_frame, text="Auto-Deletion: Disabled", text_color="red")
        self.auto_delete_status_label.pack(side="right", padx=10, pady=5)

        self.found_groups_label = ctk.CTkLabel(self.status_frame, text=f"Groups Found: {self.found_groups_count}")
        self.found_groups_label.pack(side="left", padx=10, pady=5)

        self.deleted_pictures_label = ctk.CTkLabel(self.status_frame, text=f"Pictures Deleted: {self.deleted_pictures_count}")
        self.deleted_pictures_label.pack(side="left", padx=10, pady=5)

        # --- Left Panel: Duplicate List ---
        self.list_frame = ctk.CTkFrame(self)
        self.list_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.list_frame.grid_rowconfigure(0, weight=1)
        self.list_frame.grid_columnconfigure(0, weight=1)

        self.duplicate_list_scrollable_frame = ctk.CTkScrollableFrame(self.list_frame, label_text="Found Duplicates")
        self.duplicate_list_scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # --- Right Panel: Image Viewer and Controls ---
        self.viewer_frame = ctk.CTkFrame(self)
        self.viewer_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.viewer_frame.grid_columnconfigure(0, weight=1) # Single column for dynamic images
        self.viewer_frame.grid_rowconfigure(0, weight=1) # Scrollable frame for images
        self.viewer_frame.grid_rowconfigure(1, weight=0) # Buttons

        self.image_viewer_scrollable_frame = ctk.CTkScrollableFrame(self.viewer_frame, label_text="Duplicate Group")
        self.image_viewer_scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.image_viewer_scrollable_frame.grid_columnconfigure(0, weight=1)

        # Control buttons
        self.control_button_frame = ctk.CTkFrame(self.viewer_frame)
        self.control_button_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.confirm_selection_button = ctk.CTkButton(self.control_button_frame, text="Confirm Selection (Keep Selected) / Delete All (None Selected)", command=self.confirm_selection)
        self.confirm_selection_button.grid(row=0, column=0, padx=5)

        self.skip_group_button = ctk.CTkButton(self.control_button_frame, text="Skip Group", command=self.skip_group)
        self.skip_group_button.grid(row=0, column=1, padx=5)

        self.quit_button = ctk.CTkButton(self.control_button_frame, text="Quit", command=self.quit_app)
        self.quit_button.grid(row=0, column=2, padx=5) # Adjust column

        self.start_auto_delete_button = ctk.CTkButton(self.control_button_frame, text="Start Auto-Delete", command=self.start_auto_delete)
        self.start_auto_delete_button.grid(row=0, column=3, padx=5)
        self.stop_auto_delete_button = ctk.CTkButton(self.control_button_frame, text="Stop Auto-Delete", command=self.stop_auto_delete)
        self.stop_auto_delete_button.grid(row=0, column=4, padx=5)

        # Initialize auto_delete_active based on initial setting
        if self.auto_delete_initial_setting.get():
            self.start_auto_delete()

        # Selection state for click-to-keep (initialized here)
        self.selected_image_path = None
        self.selected_image_label_ref = None # Reference to the actual CTkLabel widget

        # Start worker and queue polling after main UI is set up
        self.start_worker()
        self.after(100, self.process_queue) # Start polling the queue

    def start_worker(self):
        self.worker_thread = DuplicateFinderWorker(self.directory, self.data_queue)
        self.worker_thread.start()
        self.status_label.configure(text="Starting background processing...")

    def process_queue(self):
        while True:
            try:
                msg_type, data = self.data_queue.get_nowait()
                if msg_type == "status":
                    self.status_label.configure(text=data)
                elif msg_type == "duplicate_group":
                    group_paths = data
                    # Only add groups with more than one image to the UI
                    if len(group_paths) > 1:
                        duplicate_id = len(self.all_duplicates) # Simple unique ID for the group
                        self.all_duplicates.append({"id": duplicate_id, "paths": group_paths, "status": "pending"})
                        self.found_groups_count += 1
                        self.found_groups_label.configure(text=f"Groups Found: {self.found_groups_count}")
                        
                        if self.auto_delete_active:
                            # If auto-delete is active, immediately process this group
                            self.selected_image_path = group_paths[0] # Select the first image to keep
                            self.current_pair_paths = group_paths # Set current group for confirm_selection
                            self.confirm_selection()
                            # No need to create a button as it's processed immediately
                        else:
                            # In manual mode, add to list and create button
                            button_text = f"Group {duplicate_id}: {len(group_paths)} images"
                            btn = ctk.CTkButton(self.duplicate_list_scrollable_frame, text=button_text, command=lambda id=duplicate_id: self.display_selected_group(id))
                            btn.pack(fill="x", pady=2)
                            self.duplicate_buttons[duplicate_id] = btn
                elif msg_type == "done":
                    self.status_label.configure(text="Processing complete.")
                    self.worker_thread.join() # Ensure thread finishes
                    break # Stop polling once worker is done
            except queue.Empty:
                break # No more messages in queue
            except Exception as e:
                print(f"Error processing queue message: {e}")
                self.status_label.configure(text=f"Error in GUI: {e}")
                break
        
        self.after(100, self.process_queue) # Schedule next check

    def display_selected_group(self, duplicate_id):
        print(f"DEBUG: display_selected_group called for ID: {duplicate_id}")
        group_data = next((item for item in self.all_duplicates if item["id"] == duplicate_id), None)
        if group_data:
            group_paths = group_data["paths"]
            self.current_pair_paths = group_paths # Store the entire group for processing
            
            # Clear previous images from the viewer
            for widget in self.image_viewer_scrollable_frame.winfo_children():
                widget.destroy()
            self.selected_image_path = None
            self.selected_image_label_ref = None

            # Dynamically display images
            for i, path in enumerate(group_paths):
                try:
                    img_pil = Image.open(path)
                    max_size = (300, 300) # Max size for dynamically displayed images
                    img_pil.thumbnail(max_size, Image.LANCZOS)

                    # Create CTkImage and store it on the label to prevent garbage collection
                    ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(img_pil.width, img_pil.height))
                    
                    # Create frame for each image and its path
                    image_entry_frame = ctk.CTkFrame(self.image_viewer_scrollable_frame)
                    image_entry_frame.pack(pady=10, padx=10, fill="x", expand=True)
                    image_entry_frame.grid_columnconfigure(0, weight=1)

                    image_label = ctk.CTkLabel(image_entry_frame, text="")
                    image_label.configure(image=ctk_img) # Set the image
                    image_label.image = ctk_img # Crucial: Keep a reference
                    image_label.grid(row=0, column=0, sticky="nsew")
                    
                    # Bind click event for selection
                    image_label.bind("<Button-1>", lambda event, p=path, l=image_label: self.select_image_to_keep(p, l))

                    # Path label below image
                    path_label = ctk.CTkLabel(image_entry_frame, text=Path(path).name, wraplength=280)
                    path_label.grid(row=1, column=0, pady=(5, 0))

                    img_pil.close()

                except FileNotFoundError:
                    error_text = f"File not found:\n{Path(path).name}"
                    error_label = ctk.CTkLabel(self.image_viewer_scrollable_frame, text=error_text, wraplength=280)
                    error_label.pack(pady=10)
                except Exception as e:
                    error_text = f"Error loading image {Path(path).name}: {e}"
                    error_label = ctk.CTkLabel(self.image_viewer_scrollable_frame, text=error_text, wraplength=280)
                    error_label.pack(pady=10)

        else:
            print(f"Error: Duplicate group with ID {duplicate_id} not found.")

    def select_image_to_keep(self, image_path, label_widget):
        # Clear previous selection highlight
        if self.selected_image_label_ref:
            self.selected_image_label_ref.configure(fg_color="transparent") # Or default color

        # Apply highlight to new selection
        label_widget.configure(fg_color="blue") # Example highlight color
        self.selected_image_path = image_path
        self.selected_image_label_ref = label_widget
        print(f"Selected image to keep: {image_path}")

    def confirm_selection(self):
        if not self.current_pair_paths:
            print("No group selected to confirm.")
            return

        group_paths = self.current_pair_paths
        processed_group_id = None

        if not self.selected_image_path:
            # No image selected to keep, delete all in the current group
            deleted_count = 0
            for path_to_delete in group_paths:
                try:
                    os.remove(path_to_delete)
                    print(f"Deleted: {path_to_delete}")
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {path_to_delete}: {e}")
            print(f"No image selected to keep. Deleted all {deleted_count} images in the current group.")
            self.deleted_pictures_count += deleted_count
            self.deleted_pictures_label.configure(text=f"Pictures Deleted: {self.deleted_pictures_count}")
            
            # Find the group data to mark as processed
            for item in self.all_duplicates:
                if item["paths"] == group_paths:
                    processed_group_id = item["id"]
                    item["status"] = "all_deleted"
                    break
            
            if processed_group_id is not None:
                self.update_group_status_and_remove_button(processed_group_id, "completed")
            else:
                print("Error: Current group not found in duplicate list after all deletions.")
            
            self.clear_viewer_and_selection()
            return

        # If an image is selected, proceed with the original logic
        # Find the group data
        for item in self.all_duplicates:
            if item["paths"] == group_paths:
                processed_group_id = item["id"]
                item["status"] = "kept_one"
                break
        
        if processed_group_id is None:
            print("Error: Current group not found in duplicate list.")
            return

        # Delete unselected images
        deleted_count = 0
        for path_to_delete in group_paths:
            if path_to_delete != self.selected_image_path:
                try:
                    os.remove(path_to_delete)
                    print(f"Deleted: {path_to_delete}")
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {path_to_delete}: {e}")
                    # Optionally update item status to reflect partial deletion or error
        
        print(f"Confirmed selection: Kept {self.selected_image_path}. Deleted {deleted_count} others.")
        self.deleted_pictures_count += deleted_count
        self.deleted_pictures_label.configure(text=f"Pictures Deleted: {self.deleted_pictures_count}")
        self.update_group_status_and_remove_button(processed_group_id, "completed")
        self.clear_viewer_and_selection()

    def skip_group(self):
        if not self.current_pair_paths:
            print("No group selected to skip.")
            return
        
        group_paths = self.current_pair_paths
        processed_group_id = None

        # Find the group data
        for item in self.all_duplicates:
            if item["paths"] == group_paths:
                processed_group_id = item["id"]
                item["status"] = "skipped"
                break

        if processed_group_id is None:
            print("Error: Current group not found in duplicate list.")
            return

        print(f"Skipped group: {group_paths}")
        self.update_group_status_and_remove_button(processed_group_id, "skipped")
        self.clear_viewer_and_selection()


    def update_group_status_and_remove_button(self, group_id, status_text):
        # Update status (already done in confirm/skip, but good to have a dedicated fn if needed)
        # Remove the button from the GUI list
        if group_id in self.duplicate_buttons:
            btn = self.duplicate_buttons[group_id]
            btn.pack_forget()
            del self.duplicate_buttons[group_id]
        
        # Remove the group from all_duplicates list
        self.all_duplicates = [d for d in self.all_duplicates if d["id"] != group_id]


    def clear_viewer_and_selection(self):
        # Clear images and path labels from the viewer
        for widget in self.image_viewer_scrollable_frame.winfo_children():
            widget.destroy()
        
        # Clear selection state
        self.selected_image_path = None
        self.selected_image_label_ref = None
        self.current_pair_paths = None
        self.update_idletasks()
        print("Viewer cleared and selection reset.")


    # Commenting out process_choice as it will be completely replaced
    # def process_choice(self, choice):
    #     print(f"DEBUG: Entering process_choice for choice: {choice}")
    #     if not self.current_pair_paths:
    #         print("No pair selected to process.")
    #         return

    #     path1, path2 = self.current_pair_paths
        
    #     # Find the processed duplicate item in our list
    #     processed_duplicate_id = None
    #     for i, item in enumerate(self.all_duplicates):
    #         if item["paths"] == (path1, path2):
    #             processed_duplicate_id = item["id"]
    #             if choice == "delete_right":
    #                 item["status"] = "deleted_right"
    #                 try:
    #                     os.remove(path2)
    #                     print(f"Deleted: {path2}")
    #                 except OSError as e:
    #                     print(f"Error deleting {path2}: {e}")
    #                     item["status"] = "deletion_failed"
    #             elif choice == "delete_left":
    #                 item["status"] = "deleted_left"
    #                 try:
    #                     os.remove(path1)
    #                     print(f"Deleted: {path1}")
    #                 except OSError as e:
    #                     print(f"Error deleting {path1}: {e}")
    #                     item["status"] = "deletion_failed"
    #             elif choice == "skip":
    #                 item["status"] = "skipped"
    #                 print("Skipped pair.")
                
    #             # Update the button appearance/state and remove from future display
    #             if processed_duplicate_id is not None and processed_duplicate_id in self.duplicate_buttons:
    #                 btn = self.duplicate_buttons[processed_duplicate_id]
    #                 btn.pack_forget()  # Remove from layout
    #                 del self.duplicate_buttons[processed_duplicate_id]
                
    #             # Remove the item from self.all_duplicates
    #             self.all_duplicates = [d for d in self.all_duplicates if d["id"] != processed_duplicate_id]
    #             break

    #     self.image_label_left.configure(image=self.blank_ctk_image, text="")
    #     self.image_label_right.configure(image=self.blank_ctk_image, text="")
    #     self.path_label_left.configure(text="")
    #     self.path_label_right.configure(text="")
    #     self.current_pair_paths = None # Clear current selection

    #     # Explicitly clear CTkImage references
    #     self.ctk_img1 = None
    #     self.ctk_img2 = None

    #     self.update_idletasks() # Force GUI update

    #     # After processing a pair, if there are still duplicates, or if worker is still running
    #     # and no more duplicates, just update status if all are processed.
    #     if not self.all_duplicates and (not self.worker_thread or not self.worker_thread.is_alive()):
    #         self.status_label.configure(text="All duplicates processed.")


    def quit_app(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.stop_event.set() # Signal worker to stop
            self.worker_thread.join(timeout=5) # Wait for worker to finish (with timeout)
            if self.worker_thread.is_alive():
                print("Warning: Worker thread did not terminate gracefully.")
        self.destroy()

    def start_auto_delete(self):
        self.auto_delete_active = True
        self.auto_delete_status_label.configure(text="Auto-Deletion: Enabled", text_color="green")
        print("Auto-deletion enabled.")

        # Process any pending duplicate groups that were found while auto-delete was off
        # Iterate over a copy of the list to allow modification during iteration
        for item in list(self.all_duplicates): # Create a copy to iterate while modifying the original list
            if item["status"] == "pending":
                # Simulate processing as if it just came from the queue
                self.selected_image_path = item["paths"][0] # Select the first image to keep
                self.current_pair_paths = item["paths"] # Set current group for confirm_selection
                self.confirm_selection()
                # The button for this group will be removed by confirm_selection

    def stop_auto_delete(self):
        self.auto_delete_active = False
        self.auto_delete_status_label.configure(text="Auto-Deletion: Disabled", text_color="red")
        print("Auto-deletion disabled.")


def find_image_files(directory_path):
    image_paths = []
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
    for filepath in Path(directory_path).rglob('*'):
        if filepath.is_file() and filepath.suffix.lower() in extensions:
            image_paths.append(filepath)
    return image_paths

def compute_hashes(image_paths):
    hashes = {}
    for path in image_paths:
        try:
            img = Image.open(path)
            hash_val = imagehash.phash(img)
            img.close() # Close image to free memory
            if hash_val in hashes:
                hashes[hash_val].append(path)
            else:
                hashes[hash_val] = [path]
        except Exception as e:
            print(f"Could not process {path}: {e}")
    return {k: v for k, v in hashes.items() if len(v) > 1}

# This function is now mostly integrated into the worker's run method
# def process_image_groups(image_groups, model, threshold=0.98):
#     all_duplicate_pairs = []
#     for group_paths in image_groups.values():
#         images = []
#         valid_group_paths = []
#         for p in group_paths:
#             try:
#                 img = Image.open(p)
#                 images.append(img)
#                 valid_group_paths.append(p)
#             except Exception as e:
#                 print(f"Could not open image {p} for embedding: {e}")
        
#         if not images:
#             continue

#         embeddings = model.encode(images, convert_to_tensor=True)
#         for img in images:
#             img.close() # Close images after encoding

#         cos_scores = util.cos_sim(embeddings, embeddings)
#         for i in range(len(cos_scores) - 1):
#             for j in range(i + 1, len(cos_scores)):
#                 if cos_scores[i][j] >= threshold:
#                     all_duplicate_pairs.append((valid_group_paths[i], valid_group_paths[j]))
#     return all_duplicate_pairs

# This function is now replaced by the GUI logic
# def handle_duplicate_pairs_gui(duplicate_pairs):
#     if not duplicate_pairs:
#         print("No duplicate pairs to display.")
#         return

#     app = DuplicateViewerApp(duplicate_pairs)
#     app.mainloop()

if __name__ == "__main__":
    app = DuplicateFinderApp()
    app.mainloop()

    print("\nDuplicate finding process completed.")
