use std::collections::HashMap;
use std::fs::{self, DirEntry};
use std::path::Path;
use std::env;

fn rename_files_in_folder_for_yolov4(folder: &str) {
    let mut entries: Vec<DirEntry> = fs::read_dir(folder)
        .expect("Failed to read directory")
        .filter_map(|entry| entry.ok())
        .collect();
    
    // Sort to ensure consistent numbering
    entries.sort_by_key(|e| e.path());
    
    let mut img_count = 1;
    
    for entry in entries {
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "jpg" || ext == "png" || ext == "jpeg" { 
                let img_stem = path.file_stem().unwrap().to_str().unwrap();
                let txt_file = path.with_extension("txt");
                
                if txt_file.exists() {
                    let new_img_name = format!("img_{:04}.{}", img_count, ext.to_str().unwrap());
                    let new_txt_name = format!("img_{:04}.txt", img_count);
                    let new_img_path = path.with_file_name(&new_img_name);
                    let new_txt_path = txt_file.with_file_name(&new_txt_name);
                    
                    fs::rename(&path, &new_img_path).expect("Failed to rename image file");
                    fs::rename(&txt_file, &new_txt_path).expect("Failed to rename annotation file");
                    
                    println!("Renamed: {:?} -> {:?}, {:?} -> {:?}", path, new_img_path, txt_file, new_txt_path);
                    img_count += 1;
                }
            }
        }
    }
}

use std::{
    collections::{ HashSet},
    path::{ PathBuf},
};

/// Scan a folder for images and plan renames; returns (image_plan, label_name_map)
fn plan_image_renames(folder: &Path) -> (Vec<(PathBuf, PathBuf)>, HashMap<String, String>) {
    // 1. Gather image files
    let mut images: Vec<PathBuf> = fs::read_dir(folder)
        .expect("read_dir failed")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e.to_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                .unwrap_or(false)
        })
        .collect();

    // 2. Sort for deterministic ordering
    images.sort_by_key(|p| p.file_name().unwrap().to_os_string());

    let mut image_plan = Vec::with_capacity(images.len());
    let mut txt_name_map = HashMap::with_capacity(images.len());
    let mut target_paths = HashSet::with_capacity(images.len());

    // 3. Build plan, zero‑pad to 4 digits
    for (i, old_path) in images.into_iter().enumerate() {
        let idx = i + 1;
        let ext = old_path.extension().and_then(|e| e.to_str()).unwrap();
        let new_img_name = format!("img_{:04}.{}", idx, ext);
        let new_txt_name = format!("img_{:04}.txt", idx);

        let new_img_path = folder.join(&new_img_name);
        let new_txt_key = old_path.with_extension("txt").file_name().unwrap().to_str().unwrap().to_string();

        // Check for duplicate target path
        if !target_paths.insert(new_img_path.clone()) {
            panic!("Target image path {:?} is duplicated!", new_img_path);
        }

        image_plan.push((old_path.clone(), new_img_path));
        txt_name_map.insert(new_txt_key, new_txt_name);
    }

    (image_plan, txt_name_map)
}

/// Scan a folder for .txt labels and plan renames based on the map
fn plan_label_renames(folder: &Path, name_map: &HashMap<String, String>) -> Vec<(PathBuf, PathBuf)> {
    let mut txts: Vec<PathBuf> = fs::read_dir(folder)
        .expect("read_dir failed")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
        .collect();

    txts.sort_by_key(|p| p.file_name().unwrap().to_os_string());

    let mut plan = Vec::with_capacity(txts.len());
    let mut target_paths = HashSet::with_capacity(txts.len());

    for old_path in txts {
        let old_name = old_path.file_name().unwrap().to_str().unwrap();
        if let Some(new_name) = name_map.get(old_name) {
            let new_path = folder.join(new_name);
            if !target_paths.insert(new_path.clone()) {
                panic!("Target label path {:?} is duplicated!", new_path);
            }
            plan.push((old_path.clone(), new_path));
        } else {
            eprintln!("Warning: no mapping for label {:?}", old_name);
        }
    }
    plan
}

/// Walks `source_folder`, finds `test/images` and `test/labels`, and renames them.
pub fn rename_yolov8(source_folder: &str) {
    let root = Path::new(source_folder);
    for entry in fs::read_dir(root).expect("read_dir failed").filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() && (path.file_name().unwrap() == "test" || path.file_name().unwrap() == "train" || path.file_name().unwrap() == "valid") {
            let img_dir = path.join("images");
            let lbl_dir = path.join("labels");

            // Build plans
            let (img_plan, txt_map) = plan_image_renames(&img_dir);
            let lbl_plan = plan_label_renames(&lbl_dir, &txt_map);

            // Execute image renames
            for (old, new) in img_plan {
                fs::rename(&old, &new)
                    .unwrap_or_else(|e| panic!("Failed to rename image {:?} -> {:?}: {}", old, new, e));
                println!("Image: {:?} -> {:?}", old.file_name().unwrap(), new.file_name().unwrap());
            }

            // Execute label renames
            for (old, new) in lbl_plan {
                fs::rename(&old, &new)
                    .unwrap_or_else(|e| panic!("Failed to rename label {:?} -> {:?}: {}", old, new, e));
                println!("Label: {:?} -> {:?}", old.file_name().unwrap(), new.file_name().unwrap());
            }
        }
    }
}

fn process_yolo_v8_split(base: &str) -> Result<(), std::io::Error> {
    let source_base = Path::new(base);
    for end in ["train", "valid", "test"] {
        let base = source_base.join(end);
        if !base.is_dir() {
            continue;
        }
        let imgs = base.clone().join("images");
        let lbls = base.clone().join("labels");

        fs::create_dir_all(&imgs)?;
        fs::create_dir_all(&lbls)?;

        for entry in fs::read_dir(&base)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    let fname = entry.file_name();
                    let lowercase = ext.to_lowercase();
                    if ["jpg", "jpeg", "png"].contains(&lowercase.as_str()) {
                        fs::rename(&path, imgs.join(&fname))?;
                        println!("Moved image {:?}", fname);
                    } else if lowercase == "txt" {
                        fs::rename(&path, lbls.join(&fname))?;
                        println!("Moved label {:?}", fname);
                    }
                }
            }
        }
    }
    Ok(())
}

fn main() {
    let mut folders: Vec<String> = env::args().skip(1).collect();

    if folders.is_empty() {
        eprintln!("Usage: cargo run -- <folder1> [<folder2> ...] [mode]");
        eprintln!("  mode (optional, last arg): 'v4' (default) or 'v8'");
        return;
    }

    // Check last argument for mode tokens
    let mut is_yolo_v4: bool = true; // default
    if let Some(last) = folders.last().map(|s| s.as_str()) {
        match last {
            "v8" | "--v8" => {
                is_yolo_v4 = false;
                folders.pop();
            }
            "v4" | "--v4" => {
                folders.pop();
            }
            _ => {
                // last arg is not a mode token -> treat all args as folders, which is default behaviour
            }
        }
    }

    if folders.is_empty() {
        eprintln!("Error: no folder provided. Provide at least one folder path before the optional mode.");
        eprintln!("Example: cargo run -- \"C:\\datasets\\set1\" v8");
        return;
    }

    println!("Mode: {}", if is_yolo_v4 { "YOLOv4" } else { "YOLOv8" });
    for folder in folders {
        println!("Processing folder: {}", folder);
        if is_yolo_v4 {
            rename_files_in_folder_for_yolov4(&folder);
        } else {
            process_yolo_v8_split(&folder).unwrap_or_else(|e| {
                eprintln!("Error processing folder {}: {}", folder, e);
            });
            rename_yolov8(&folder);
        }
    }
}

