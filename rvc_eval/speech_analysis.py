import whisper
import json

def analyze_audio(filename):
    mysp = __import__("my-voice-analysis")

    # Get directory name
    directory = os.path.dirname(filename)
    # Get file name
    file = os.path.basename(filename)
    file = file.replace('.wav','')

    print("filename: ", filename)
    print("file: ", file)
    print("directory: ", directory)
    
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    text = result["text"]
    print('text: ', text)
    print('')
    
    gender, emotion = mysp.myspgend(file, directory)
    dataset,number_of_syllables,number_of_pauses,rate_of_speech,articulation_rate,speaking_duration,original_duration,balance,f0_mean,f0_std,f0_median,f0_min,f0_max,f0_quantile25,f0_quan75 = mysp.mysptotal(file, directory)
    
    data = {
        "filename": filename,
        "text": text,
        "gender": gender,
        "emotion": emotion,
        "number_of_syllables": number_of_syllables,
        "number_of_pauses": number_of_pauses,
        "rate_of_speech": rate_of_speech,
        "articulation_rate": articulation_rate,
        "speaking_duration": speaking_duration,
        "original_duration": original_duration,
        "balance": balance,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_median": f0_median,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "f0_quantile25": f0_quantile25,
        "f0_quan75": f0_quan75
    }
    
    # Save the data dictionary to a JSON file
    json_filename = file + "_analysis.json"
    with open(json_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return json_filename  # Return the name of the json file that has been created

if __name__ == "__main__":
    # Test the function
    test_filename = "C:/Users/input/voice_test.wav"
    result_json = analyze_audio(test_filename)
    print(f"Analysis saved in: {result_json}")
