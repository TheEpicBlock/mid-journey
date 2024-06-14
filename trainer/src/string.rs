use crate::{input::Config, layer::Size, training_data::GpuInputData};

pub(crate) fn get_input_size(configured_input_length: Size) -> Size {
    return configured_input_length * 27;
}

pub fn string_to_data(str: &str, config: &Config) -> GpuInputData {
    let mut output = vec![0.0; config.input_length() as usize];

    let mut words: Vec<_> = str.split_whitespace().collect();
    let last_word = words.iter().enumerate().filter(|w| !w.1.starts_with("(")).last().map(|l| l.0);
    let last_word = last_word.map(|i| words.remove(i));

    for (i, char) in itertools::join(words, " ").chars().enumerate() {
        if let Some(n) = char_to_num(char) {
            output[i * 27 + n] = 1.0;
        }
    }
    if let Some(last_word) = last_word {
        for (i, char) in last_word.chars().enumerate() {
            if let Some(n) = char_to_num(char) {
                output[(config.input_length_max_chars() as usize - i as usize - 1) * 27 + n] = 1.0;
            }
        }
    }

    return output;
}

fn char_to_num(c: char) -> Option<usize> {
    let char = c.to_ascii_lowercase();
    if !char.is_whitespace() {
        if char.is_ascii() && char.is_alphabetic() {
            Some((char as u8 - 'a' as u8) as usize)
        } else {
            Some(26)
        }
    } else {
        None
    }
}