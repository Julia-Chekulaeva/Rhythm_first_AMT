This is a repository with my Master's Thesis work: "A Hybrid Rhythm-First Pipeline for Robust AutomaticTranscription of Music Audio into Musical Notation". At current moment it contains several folders:

* `modules` - folder with core Python files with code;
* `blocknotes` - folder with `.ipynb` blocknotes which were used for manual testing pipeline;
* `audio_folder` - folder with some utility audio: audio with the click sound used in the blocknote `tempo_extraction.ipynb` for manual checking of onsets correctness, and `Chroma gamma.mp3`;
* `sound_examples` - folder with some audio examples used while manually testing the algorithm;
* `musicxml_folder` - folder with some MusicXML files generated while manually testing the algorithm.

Besides that, there are some files in the root directory:

* `main.py` - file with the main function; there are also variables `PATH_TO_AUDIO` (path to the source audio) and `PATH_TO_MXML` (path to the destination MusicXML file) which should be specified before launching it;
* `Example_diploma.musicxml` - an example for the audio `Example_diploma.mp3`.

Folder `modules` contains files:

* `get_musicxml.py` - file with functions for MusicXML generation;
* `global_vars.py` - file with global variables;
* `help_funcs.py`- file with the help functions;
* `notes_extract.py` - file with functions for pitch estimation;
* `rythm_extract.py` - file with functions for rythm extraction.
