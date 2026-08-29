
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, HORIZONTAL, filedialog
from tkinter import ttk
import pandas as pd

# Use the consolidated pipeline from silic2 (now supports multiprocessing)
from silic2 import browser

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_CLASS_COLUMNS = ['soundclass_id', 'species_name', 'sound_class', 'scientific_name']


def read_target_class_ids(csv_path):
    """Read target IDs, accepting both the public and legacy model column names."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    id_column = next(
        (column for column in ('soundclass_id', 'soundclassid', 'sounclass_id') if column in df.columns),
        None,
    )
    if id_column is None:
        raise ValueError('CSV must contain a soundclass_id column.')
    ids = []
    for value in df[id_column].dropna():
        try:
            ids.append(str(int(value)))
        except (TypeError, ValueError):
            raise ValueError(f'Invalid soundclass_id: {value}') from None
    return list(dict.fromkeys(ids))


def write_target_classes(csv_path, records):
    pd.DataFrame(records, columns=TARGET_CLASS_COLUMNS).to_csv(
        csv_path,
        index=False,
        encoding='utf-8-sig',
    )

def main():
    root = tk.Tk()
    root.title('SILIC 2 - Sound Identification and Labeling Intelligence for Creatures V2')
    try:
        root.iconbitmap(os.path.join(PROJECT_DIR, 'model', 'LOGO_circle.ico'))
    except Exception:
        pass

    inputfolder = tk.StringVar(root)
    outputfolder = tk.StringVar(root)
    threshold = tk.DoubleVar(root)
    model_weight = tk.StringVar(root)
    workers = tk.IntVar(root, value=1)
    event_queue = queue.Queue()

    w = 800
    h = 580
    x = round((root.winfo_screenwidth()-w)/2)
    y = round((root.winfo_screenheight()-h)/2)
    root.geometry(f"{w}x{h}+{x}+{y}")

    def select_input_folder():
        path = filedialog.askdirectory()
        inputfolder.set(path)
        folder_input_path_label.config(text=path)

    def select_output_folder():
        path = filedialog.askdirectory()
        outputfolder.set(path)
        folder_output_path_label.config(text=path)

    def filter_options(event):
        text_val = filter_entry.get()
        listbox2.delete(0, tk.END)
        for option in options:
            if option.find(text_val) >= 0:
                listbox2.insert(tk.END, option)

    def shift_selection(event):
        selected = event.widget.curselection()
        source = event.widget
        destination = listbox2 if source == listbox1 else listbox1
        # collect indices first (avoid shifting during iteration)
        selected_indices = list(selected)[::-1]
        for i in selected_indices:
            item = source.get(i)
            destination.insert(tk.END, item)
            source.delete(i)

    def setThreshold(source):
        threshold.set(thresholdSlider.get())

    def setModel(source):
        nonlocal classes, options
        try:
            classes = readclassfile()
        except (OSError, KeyError, pd.errors.ParserError) as exc:
            messagebox.showerror('Model Selection', f'Cannot load model classes:\n{exc}')
            return
        listbox1.delete(0, tk.END)
        listbox2.delete(0, tk.END)
        options = list(classes)
        messagebox.showinfo('Model Selection', 'Model %s was selected including %s sound classes.' %(model_weight.get(), len(options)))
        for item in options:
            listbox2.insert(tk.END, item)

    def readclassfile():
        classes = {}
        class_path = os.path.join(PROJECT_DIR, 'model', model_weight.get(), 'soundclass.csv')
        df = pd.read_csv(class_path, lineterminator='\n', encoding="utf-8")
        df = df.sort_values(by=['species_name', 'sound_class'])
        for _, row in df.iterrows():
            key = f"{row['sounclass_id']}: {row['species_name']}({row['scientific_name']}) {row['sound_class']}"
            classes[key] = {
                'sounclass_id': row['sounclass_id'],
                'species_name': row['species_name'],
                'sound_class': row['sound_class'],
                'scientific_name': row['scientific_name']
            }
        return classes

    def export_target_classes():
        selected = listbox1.get(0, tk.END)
        if not selected:
            messagebox.showwarning('Export Target Classes', 'No target sound classes selected.')
            return
        csv_path = filedialog.asksaveasfilename(
            title='Export Target Sound Classes',
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv')],
            initialfile='target_soundclasses.csv',
        )
        if not csv_path:
            return
        records = []
        for item in selected:
            sound_class = classes[item]
            records.append({
                'soundclass_id': sound_class['sounclass_id'],
                'species_name': sound_class['species_name'],
                'sound_class': sound_class['sound_class'],
                'scientific_name': sound_class['scientific_name'],
            })
        try:
            write_target_classes(csv_path, records)
        except OSError as exc:
            messagebox.showerror('Export Target Classes', f'Cannot write CSV:\n{exc}')
            return
        messagebox.showinfo('Export Target Classes', f'Exported {len(records)} sound classes.')

    def import_target_classes():
        csv_path = filedialog.askopenfilename(
            title='Import Target Sound Classes',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
        )
        if not csv_path:
            return
        try:
            imported_ids = set(read_target_class_ids(csv_path))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            messagebox.showerror('Import Target Classes', f'Cannot read CSV:\n{exc}')
            return

        id_to_item = {
            str(int(sound_class['sounclass_id'])): item
            for item, sound_class in classes.items()
        }
        matched_ids = [soundclass_id for soundclass_id in imported_ids if soundclass_id in id_to_item]
        missing_ids = sorted(imported_ids.difference(id_to_item), key=lambda value: int(value))
        listbox1.delete(0, tk.END)
        listbox2.delete(0, tk.END)
        matched_items = {id_to_item[soundclass_id] for soundclass_id in matched_ids}
        for item in options:
            destination = listbox1 if item in matched_items else listbox2
            destination.insert(tk.END, item)

        message = f'Imported {len(matched_items)} target sound classes.'
        if missing_ids:
            message += f"\n{len(missing_ids)} IDs are not in the current model: {', '.join(missing_ids)}"
        messagebox.showinfo('Import Target Classes', message)

    # Progress state
    progress_total = tk.IntVar(root, value=0)
    progress_done = tk.IntVar(root, value=0)
    run_start_ts = tk.DoubleVar(root, value=0.0)

    def _fmt_hhmmss(sec: float) -> str:
        sec = int(max(0, sec))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        else:
            return f"{m:02d}:{s:02d}"

    def gui_log(msg: str):
        event_queue.put(('log', msg))

    def gui_progress(done:int, total:int):
        event_queue.put(('progress', int(done), int(total)))

    def apply_progress(done, total):
        # Initialize maximum when total changes
        if progress_total.get() != int(total):
            progress_total.set(int(total))
            pbar.configure(maximum=max(1, int(total)))
        progress_done.set(int(done))
        pbar['value'] = int(done)
        pct = 0 if total == 0 else int(done*100/total)
        plabel.config(text=f"{done}/{total} ({pct}%)")
        # 時間：已花費與預估完成
        now = time.time()
        start = run_start_ts.get() or now
        elapsed = now - start
        # ETA: 依照目前速率（每檔平均時間）估算總時間
        if done > 0 and total > 0:
            avg = elapsed / max(1, done)
            est_total = avg * total
            eta_remain = max(0, est_total - elapsed)
            tlabel.config(text=f"{_fmt_hhmmss(elapsed)} / {_fmt_hhmmss(elapsed + eta_remain)}")
        else:
            tlabel.config(text=f"{_fmt_hhmmss(0)} / --:--")

    def poll_events():
        try:
            while True:
                event = event_queue.get_nowait()
                if event[0] == 'log':
                    text.insert(tk.END, event[1] + "\n")
                    text.see(tk.END)
                elif event[0] == 'progress':
                    apply_progress(event[1], event[2])
                elif event[0] == 'finished':
                    run_button.config(state=tk.NORMAL)
                elif event[0] == 'error':
                    run_button.config(state=tk.NORMAL)
                    messagebox.showerror('SILIC Error', event[1])
        except queue.Empty:
            pass
        root.after(100, poll_events)

    def run():
        # Validate paths
        if not inputfolder.get():
            messagebox.showwarning('Warning','No input folder found.')
            return
        if not outputfolder.get():
            messagebox.showwarning('Warning','No output folder found.')
            return

        # Build target class filter
        selected_ids = []
        if listbox1.get(0, tk.END):
            for item in listbox1.get(0, tk.END):
                selected_ids.append(str(classes[item]['sounclass_id']))

        # Clear log and reset progress bar
        text.delete("1.0", tk.END)
        progress_total.set(0)
        progress_done.set(0)
        pbar.configure(maximum=1)
        pbar['value'] = 0
        plabel.config(text="0/0 (0%)")
        tlabel.config(text="00:00 / --:--")
        run_start_ts.set(time.time())
        run_button.config(state=tk.DISABLED)
        kwargs = {
            'source': inputfolder.get(),
            'model': model_weight.get(),
            'step': 1500,
            'targetclasses': selected_ids,
            'conf_thres': threshold.get(),
            'savepath': outputfolder.get(),
            'workers': max(1, int(workers.get())),
            'ui_callback': gui_log,
            'progress_cb': gui_progress,
        }

        def worker():
            try:
                browser(**kwargs)
            except Exception as exc:
                event_queue.put(('error', str(exc)))
            else:
                event_queue.put(('finished',))

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------
    # Layout
    # -------------------------
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=2, minsize=240)
    root.columnconfigure(2, weight=1)
    root.columnconfigure(3, weight=2, minsize=240)

    try:
        logo = tk.PhotoImage(file=os.path.join(PROJECT_DIR, 'model', 'silic_logo_full.png'))
        logo_label = tk.Label(root, image=logo)
        logo_label.image = logo  # prevent garbage collection
        logo_label.grid(row=0,column=0,rowspan=2,sticky=tk.N+tk.S+tk.W+tk.E)
    except Exception:
        pass

    select_input_folder_button = tk.Button(root, text="Select Input Folder", command=select_input_folder)
    select_input_folder_button.grid(row=0,column=1,sticky=tk.N+tk.S+tk.W+tk.E)

    folder_input_path_label = tk.Label(root, text="", anchor='w')
    folder_input_path_label.grid(row=0,column=2,columnspan=2, sticky=tk.E+tk.W)

    select_output_folder_button = tk.Button(root, text="Select Output Folder", command=select_output_folder)
    select_output_folder_button.grid(row=1,column=1,sticky=tk.E+tk.W)

    folder_output_path_label = tk.Label(root, text="", anchor='w')
    folder_output_path_label.grid(row=1,column=2,columnspan=2,sticky=tk.E+tk.W)

    # --- Controls row (all in the same row) ---
    controls = tk.Frame(root)
    controls.grid(row=2, column=0, columnspan=5, sticky=tk.E+tk.W, pady=4)
    # make slider column expand
    controls.columnconfigure(1, weight=2)
    controls.columnconfigure(3, weight=1)
    controls.columnconfigure(5, weight=0)

    # Confidence Threshold
    thresholdSlider_label = tk.Label(controls, text="Confidence Threshold", anchor='w')
    thresholdSlider_label.grid(row=0, column=0, sticky=tk.W, padx=(0,4))

    thresholdSlider = tk.Scale(controls, from_=0.0, to=1.0, resolution=0.01, length=220,
                            orient=HORIZONTAL, command=setThreshold)
    thresholdSlider.grid(row=0, column=1, sticky=tk.E+tk.W)
    thresholdSlider.set(0.1)

    # Model version
    model_label = tk.Label(controls, text="Model version", anchor='w')
    model_label.grid(row=0, column=2, sticky=tk.W, padx=(12,4))

    model_dir = os.path.join(PROJECT_DIR, 'model')
    sets = [
        item for item in os.listdir(model_dir)
        if os.path.isfile(os.path.join(model_dir, item, 'best.pt'))
        and os.path.isfile(os.path.join(model_dir, item, 'soundclass.csv'))
    ] if os.path.isdir(model_dir) else []
    sets.sort()
    latestmodel = max(
        sets,
        key=lambda item: os.path.getmtime(os.path.join(model_dir, item, 'best.pt')),
        default='',
    )
    model_weight.set(latestmodel if latestmodel else '')
    opm=tk.OptionMenu(controls, model_weight, *(sets or ['']), command=setModel)
    opm.grid(row=0, column=3, sticky=tk.E+tk.W)
    
    # Processes
    workers_label = tk.Label(controls, text="Processes", anchor='w')
    workers_label.grid(row=0, column=4, sticky=tk.W, padx=(12,4))

    workers_spin = tk.Spinbox(controls, from_=1, to=max(1, (os.cpu_count() or 2)),
                            textvariable=workers, width=6)
    workers_spin.grid(row=0, column=5, sticky=tk.W)

    target_file_controls = tk.Frame(root)
    target_file_controls.grid(row=3, column=0, columnspan=5, sticky=tk.E+tk.W, pady=(0, 4))
    import_target_button = tk.Button(
        target_file_controls,
        text='Import Target Classes CSV',
        command=import_target_classes,
    )
    import_target_button.pack(side=tk.LEFT, padx=2)
    export_target_button = tk.Button(
        target_file_controls,
        text='Export Target Classes CSV',
        command=export_target_classes,
    )
    export_target_button.pack(side=tk.LEFT, padx=2)

    # Populate class lists
    classes = readclassfile() if latestmodel else {}

    target = tk.Label(root, text="Target Classes (Left empty when detect all classes)")
    target.grid(row=4,column=0,columnspan=2,sticky=tk.E+tk.W)

    filter_label = tk.Label(root, text="Class filter")
    filter_label.grid(row=4,column=2,sticky=tk.E+tk.W)

    filter_entry = tk.Entry(root)
    filter_entry.bind('<KeyRelease>', filter_options)
    filter_entry.grid(row=4,column=3)

    listbox1 = tk.Listbox(root, height=18)
    listbox1.bind('<Double-Button-1>', shift_selection)
    listbox1.grid(row=5,column=0,columnspan=2,sticky=tk.E+tk.W)

    listbox2 = tk.Listbox(root, height=18)
    listbox2.bind('<Double-Button-1>', shift_selection)
    listbox2.grid(row=5,column=2,columnspan=2,sticky=tk.E+tk.W)

    # Populate the right listbox with some data
    options = list(classes)
    for item in options:
        listbox2.insert(tk.END, item)

    run_button = tk.Button(root, text="RUN",bg='#8BC440',fg="#000000", relief="raised", command=run)
    run_button.grid(row=6,column=0,columnspan=5,sticky=tk.E+tk.W)

    # Progress bar row
    plabel = tk.Label(root, text="0/0 (0%)", anchor='w')
    plabel.grid(row=7, column=0, sticky=tk.W, padx=2)
    pbar = ttk.Progressbar(root, orient='horizontal', mode='determinate')
    pbar.grid(row=7, column=1, columnspan=3, sticky=tk.E+tk.W, pady=4)
    # 新增時間標籤（已花費/預估完成）
    tlabel = tk.Label(root, text="00:00 / --:--", anchor='e')
    tlabel.grid(row=7, column=4, sticky=tk.E, padx=4)

    text = tk.Text(root, height=8)
    text.grid(row=8,column=0,columnspan=5,sticky=tk.E+tk.W)

    root.after(100, poll_events)
    root.mainloop()

if __name__ == '__main__':
    main()
