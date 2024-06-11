fn main() {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 3 {
        println!("Usage: {:?} <parameters> <nn_config>", args[0]);
        return;
    }
}
