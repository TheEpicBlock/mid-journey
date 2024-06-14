use std::{env, fs::File, io::{self, stdout, Write}, path::PathBuf, thread::panicking};

use crossterm::{cursor, event::{self, read, Event, KeyModifiers}, queue, style::{self, Stylize}, terminal::{disable_raw_mode, enable_raw_mode}, tty::IsTty, ExecutableCommand};
use futures::executor;
use trainer::{color::Color, gpu::init_gpu, input::{Config, JsonNetworkParameters}, layer, neural_network::{eval_single, EvalResources}};

#[tokio::main]
async fn main() {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 3 {
        println!("Usage: {:?} <parameters> <nn_config>", args[0]);
        return;
    }

    if !stdout().is_tty() {
        println!("This tool is too over-engineered to work on anything but a tty");
        return;
    }

    let network_file = PathBuf::from(&args[1]);
    let config_file = PathBuf::from(&args[2]);

    let parameters: JsonNetworkParameters = serde_json::from_reader(File::open(network_file).expect("Can't open parameters file")).unwrap();
    let config: Config = serde_json::from_reader(File::open(config_file).expect("Can't open training data file")).unwrap();

    let gpu = init_gpu().await;
    let parameters = layer::from_json(&parameters, &config, &gpu);
    let eval_resources = EvalResources::init(&gpu, &config, &parameters, &vec![(vec![0.0; config.input_length() as usize], Color::from_oklab((0.0,0.0,0.0)))]);


    // Setup the box™
    println!();
    let mut top_bar = String::new();
    top_bar += " [";
    for _ in 0..((config.input_length_max_chars()-2).max(0)) {
        top_bar += "█";
    }
    top_bar += "]";
    println!("{}", &top_bar);
    let top_bar: &str = &top_bar;

    print!("╔");
    for _ in 0..config.input_length_max_chars() {
        print!("═");
    }
    println!("╗");
    print!("║");
    for _ in 0..config.input_length_max_chars() {
        print!(" ");
    }
    println!("║");
    print!("╚");
    for _ in 0..config.input_length_max_chars() {
        print!("═");
    }
    print!("╝");

    // Enter raw mode and start accepting input
    let mut stdout = io::stdout();

    let bottom_row = cursor::position().unwrap().1;
    let mut input_buffer = String::new();
    let mut buf_position = 0;

    let _raw_mode = RawMode::start();

    let mut update = |buf: &str, buf_position: usize| {
        queue!(stdout, cursor::MoveTo(1, bottom_row-1)).unwrap();
        for _ in 0..config.input_length_max_chars() {
            queue!(stdout, style::Print(" ")).unwrap();
        }
        queue!(stdout, cursor::MoveTo(1, bottom_row-1)).unwrap();
        queue!(stdout, style::Print(&buf)).unwrap();
        stdout.flush().unwrap();

        let c = executor::block_on(eval_single(&buf, &gpu, &config, &eval_resources));
        let rgb = c.to_rgb();
        let oklab = c.to_oklab();

        queue!(stdout, cursor::MoveTo(0, bottom_row-3)).unwrap();
        queue!(stdout, style::PrintStyledContent(top_bar.with(style::Color::Rgb {
            r: (rgb.0 * 255.0) as u8,
            g: (rgb.1 * 255.0) as u8,
            b: (rgb.2 * 255.0) as u8
        }))).unwrap();
        queue!(stdout, cursor::MoveTo(0, bottom_row-4)).unwrap();
        write!(stdout, "                               ").unwrap();
        queue!(stdout, cursor::MoveTo(0, bottom_row-4)).unwrap();
        write!(stdout, "{:.3} {:.3} {:.3} {}", oklab.0, oklab.1, oklab.2, c.to_hex()).unwrap();
        queue!(stdout, cursor::MoveTo(1 + buf_position as u16, bottom_row-1)).unwrap();
        stdout.flush().unwrap();
    };
    update(&input_buffer, buf_position);

    loop {
        match read().unwrap() {
            Event::Key(event) => {
                match event.code {
                    event::KeyCode::Delete => {
                        if buf_position != input_buffer.len() {
                            input_buffer.remove(buf_position);
                        }
                    }
                    event::KeyCode::Backspace => {
                        if buf_position != 0 {
                            buf_position -= 1;
                            input_buffer.remove(buf_position);
                        }
                    }
                    event::KeyCode::Char(c) => {
                        if event.modifiers.contains(KeyModifiers::CONTROL) {
                            if c == 'c' || c == 'd' {
                                stdout.execute(cursor::MoveTo(0, bottom_row)).unwrap();
                                println!("");
                                return;
                            }
                        }
                        input_buffer.insert(buf_position, c);
                        input_buffer.truncate(config.input_length_max_chars() as usize);
                        buf_position += 1;
                        buf_position = buf_position.min(input_buffer.len());
                    }
                    event::KeyCode::Left => {
                        buf_position = buf_position.saturating_sub(1);
                        buf_position = buf_position.max(0);
                    }
                    event::KeyCode::Right => {
                        buf_position += 1;
                        buf_position = buf_position.min(input_buffer.len());
                    }
                    _ => {}
                }
                update(&input_buffer, buf_position);
            },
            Event::Paste(str) => {
                input_buffer.insert_str(buf_position, &str);
                input_buffer.truncate(config.input_length_max_chars() as usize);
                buf_position += str.len();
                buf_position = buf_position.min(input_buffer.len());
                update(&input_buffer, buf_position);
            }
            _ => {}
        }
    }
}


/// Enables raw mode whilst this struct exists
/// Ensures that raw mode is disabled even when panicking
struct RawMode;

impl RawMode {
    pub fn start() -> Self {
        enable_raw_mode().unwrap();
        return Self;
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        let r = disable_raw_mode();
        if !panicking() {
            r.unwrap();
        }
    }
}