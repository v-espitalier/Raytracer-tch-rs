use image::{Rgba, RgbaImage, ImageResult};
use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};
use std::thread;
use std::time::Duration;
// Libs for tch-rs (gpu)
use torch_sys::IntList;
use tch::{Tensor, Device, nn::VarStore, no_grad, autocast, vision::imagenet::save_image};


/////////////////////////////////////////////////////////////////////////////////////
// Initial Atari code for the Raytracer - BASIC Atari 8 bits - (Fits in 10 lignes)
// https://bunsen.itch.io/raytrace-movie-atari-8bit-by-d-scott-williamson
/////////////////////////////////////////////////////////////////////////////////////
// 1 GR.31:SE.0,7,2:SE.1,3,8:SE.2,13,15:DI.DI(16):F.I=0TO15:REA.R:DI(I)=R:N.I:A=2
// 2 F.N=0TO191:F.M=0TO159:POK.77,0:X=0:Y=-A/25:Z=A:I=SGN(M-80.5)
// 3 U=(M-80.5)/(40*1.3):V=(N-80.5)/(80*1.3):W=1/SQR(U*U+V*V+1):U=U*W:V=V*W:G=1
// 4 E=X-I:F=Y-I:P=U*E+V*F-W*Z:D=P*P-E*E-F*F-Z*Z+1:G.5+(D<=0)*3
// 5 T=-P-SQR(D):G.6+(T<=0)*2:D.0,24,6,30,36,12,42,18,9,33,3,27,45,21,39,15
// 6 X=X+T*U:Y=Y+T*V:Z=Z-T*W:E=X-I:F=Y-I:G=Z:P=2*(U*E+V*F-W*G)
// 7 U=U-P*E:V=V-P*F:W=W+P*G:I=-I:G.4
// 8 IF V<0THEN P=(Y+2)/V:S=INT(X-U*P)+INT(Z-W*P):S=S-INT(S/2)*2:V=-V*(S/2+.3)+.2
// 9 C.3-INT(((3*16)*SQR(V)+DI((M-INT(M/4)*4)+(N-INT(N/4)*4)*4)/3)/16):PL.M,191-N
// 10 N.M:N.N:A=10*(A=-1)+(A-.1)*(A>-1):G.2
/////////////////////////////////////////////////////////////////////////////////////


// Direct port of atari BASIC code to Rust
fn raytracing_cpu_atari_like(factor_res: u32, img_file_prefix: &String) {
        
    let n_img_iter: u32 = 110;     // Initial Atari program behavior
    //let n_img_iter: u32 = 10;
    //let factor_res: u32 = 1;   // Initial resolution
    //let factor_res: u32 = 3;

    // line 1
    // GR.31: Set resolution 160x192 (system instruction)
    let cx: u32 = 160;
    let cy: u32 = 192;

    // Default color
    let pixel_black = Rgba([0, 0, 0, 255]);
    // SE.0,7,2:   Color 0 = blue / dark
    let pixel_blue = Rgba([0, 0, 255, 255]);
    // SE.1,3,8:   Color 1 = purple
    let pixel_purple = Rgba([180, 80, 96, 255]);
    // SE.2,13,15: Color 2 = yellow
    let pixel_yellow = Rgba([160, 128, 0, 255]);


    // DI.DI(16): Declare constants array, dimension 16
    // F.I=0TO15:       // Feed it
    // REA.R
    // DI(I)=R
    //N.I
    // (line below) D.0,24,6,30,36,12,42,18,9,33,3,27,45,21,39,15
    // Dithering constants (tramage)
    let di: [f32; 16] = [0., 24., 6., 30., 36., 12., 42., 18., 9., 33., 3., 27., 45., 21., 39., 15.];

    //A=2
    let mut a : f32 = 2.;

    for img_index in 0..n_img_iter {
	    let mut image : RgbaImage = RgbaImage::new(cx * 2 * factor_res, cy * factor_res);
        
        for (im_x_fact, im_y_fact, pixel) in image.enumerate_pixels_mut() {
//        for n in 0..191 {              // line 2;  F.N=0TO19
//            for m in 0..159 {            // F.M=0TO159
            let m_fact: u32 = ((im_x_fact as f32) / 2.) as u32;
            let n_fact: u32 = 192 * factor_res - im_y_fact;

            let mf : f32 = (m_fact as f32) / (factor_res as f32);
            let nf : f32 = (n_fact as f32)/ (factor_res as f32);
            let m  : u32 = mf as u32;
            let n  : u32 = nf as u32;

            // POK.77,0  :  system instruction, skipped

            let mut x : f32 = 0.;           // X=0
            let mut y : f32 = - a / 25.;    // Y=-A/25:
            let mut z : f32 = a;           // Z=A
            let mut i : f32 = -1.;          // I=SGN(M-80.5)
            if mf >= 80.5 {i = 1.;}

            let mut u : f32 = (mf - 80.5) / (40. * 1.3);    // line 3; U=(M-80.5)/(40*1.3)
            let mut v : f32 = (nf - 80.5) / (80. * 1.3);    // V=(N-80.5)/(80*1.3)
            let mut w : f32 = 1. / f32::sqrt(u * u + v * v + 1.);  // W=1/SQR(U*U+V*V+1)
            u = u * w;                                    // U=U*W
            v = v * w;                                    // V=V*W
            let mut g : f32 = 1.;                         // G=1

            let mut p: f32 = 0.;
            loop {
                let mut e : f32 = x - i;                      // line 4; E=X-I
                let mut f : f32 = y - i;                      // F=Y-I
                p = u * e + v * f - w * z;                    // P=U*E+V*F-W*Z
                let d : f32 = p * p - e * e - f * f - z * z + 1.; // D=P*P-E*E-F*F-Z*Z+1

                // goto:  G.5+(D<=0)*3   // if d<=0, goto line 8 (break)
                if d <= 0. {break;}

                let mut t : f32 = -p - f32::sqrt(d);          // line 5; T=-P-SQR(D)
                // goto: G.6+(T<=0)*2    // if t<=0, goto line 8 (break)
                if t <= 0. {break;}
                // Dithering constants defined here initially in basic: D.0,24,6,30,36,12,42,18,9,33,3,27,45,21,39,15

                x = x + t * u;  // line 6; X=X+T*U
                y = y + t * v;  // Y=Y+T*V
                z = z - t * w;  // Z=Z-T*W
                e = x - i;      // E=X-I
                f = y - i;      // F=Y-I
                g = z;          // G=Z
                p = 2. * (u * e + v * f - w * g);   // P=2*(U*E+V*F-W*G)

                u = u - p * e;  // line 7; U=U-P*E
                v = v - p * f;  // V=V-P*F
                w = w + p * g;  // W=W+P*G
                i = -i;         // I=-I

            }   // goto: G.4

            if v < 0. {p = (y + 2.) / v;}      // line 8; IF V<0THEN P=(Y+2)/V

            //let mut s : f32 = f32::floor(x - u * p) + f32::floor(z - w * p);  // S=INT(X-U*P)+INT(Z-W*P)
            let mut s : i32 = (f32::floor(x - u * p) + f32::floor(z - w * p)) as i32;  // S=INT(X-U*P)+INT(Z-W*P)

            // s = s - f32::floor(s / 2.) * 2.;   // S=S-INT(S/2)*2
            s = s % 2; // s - f32::floor(s / 2.) * 2.;   // S=S-INT(S/2)*2    // Chessboard

            v = -v * ((s as f32) / 2. + 0.3) + 0.2;    // V=-V*(S/2+.3)+.2

            // line 9 C.3-INT(((3*16)*SQR(V)+DI((M-INT(M/4)*4)+(N-INT(N/4)*4)*4)/3)/16)
            //let m_diff : usize = usize::try_from(m - ((m / 4)) * 4).unwrap();
            //let m_mod : usize = usize::try_from(m % 4).unwrap();
            let m_mod : usize = usize::try_from(m_fact % 4).unwrap();
            //let n_diff : usize = usize::try_from(n - ((n / 4)) * 4).unwrap();
            //let n_mod : usize = usize::try_from(n % 4).unwrap();
            let n_mod : usize = usize::try_from(n_fact % 4).unwrap();

            let di_index : usize = m_mod + n_mod * 4;
            // Pixel color
            let c = 3 - ( (((3. * 16.) * f32::sqrt(v) + di[di_index] / 3.) / 16.) as u32);

            // PL.M,191-N - system instruction: PLOT(M, 191 - N)
            if c == 0
            {
			    *pixel = pixel_black;
            }
            if c == 1
            {
			    *pixel = pixel_blue;
            }
            else if c == 2
            {
			    *pixel = pixel_purple;
            }
            else if c == 3
            {
			    *pixel = pixel_yellow;
            }

            //} // for m in 0..159 {            // line 10; N.M
        //} // for n in 0..191 {            // N.N  (next n)
        } // for (n, m, pixel) in image.enumerate_pixels_mut() {

        // A=10*(A=-1)+(A-.1)*(A>-1)
        if a <= -1.
            {a = 10.;}
        else
            {a = a - 0.1;}

        if (img_file_prefix.len() > 1) {
            let img_f_name : String = format!("{}{:0>3}.png", img_file_prefix, img_index);
            let res : ImageResult<()> = image.save(&img_f_name);
            println!("Saved image : {}", &img_f_name);
        }
     
    }   // goto: G.2      // for img_index in 0..n_img_iter {
} // fn raytracing_cpu_atari_like


// Internal function for multithreading version
fn raytracing_cpu_single_thread(ref_img_index_min : &i32, ref_img_index_max : &i32, ref_n_img_iter : &i32, ref_factor_res: &i32, img_file_prefix: &String) 
{
    let img_index_min: i32 = *ref_img_index_min;
    let img_index_max: i32 = *ref_img_index_max;
    let n_img_iter: i32        = *ref_n_img_iter;
    let factor_res: i32    = *ref_factor_res;
    //println!("Thread img index min,max : {}, {}", img_index_min, img_index_max);

    // line 1
    // GR.31: Set resolution 160x192 (system instruction)
    let cx: i32 = 160;
    let cy: i32 = 192;
        
    for img_index in img_index_min..img_index_max {
        //println!("Image:{}", img_index);
        // DI.DI(16): Declare constants array, dimension 16
        // F.I=0TO15:       // Feed it
        // REA.R
        // DI(I)=R
        //N.I

        //A=2
        let mut a : f32 = 2. - (img_index as f32) * 0.1 / (n_img_iter as f32) * 110.;
        if a <= -1. {a = a + 11.;}
        //let mut a : f32 = 1.;

        // One iteration = one generated image
	    let mut image : RgbaImage = RgbaImage::new((cx * 2 * factor_res) as u32, (cy * factor_res) as u32);
        
        for (im_x_fact, im_y_fact, pixel) in image.enumerate_pixels_mut() {
//        for n in 0..191 {              // line 2;  F.N=0TO19
//            for m in 0..159 {            // F.M=0TO159
            let m_fact: i32 = ((im_x_fact as f32) / 2.) as i32;
            let n_fact: i32 = 192 * factor_res - (im_y_fact as i32);

            let mf : f32 = (m_fact as f32) / (factor_res as f32);
            let nf : f32 = (n_fact as f32)/ (factor_res as f32);
            let m  : i32 = mf as i32;
            let n  : i32 = nf as i32;

            // POK.77,0  :  system instruction, skipped

            let mut x : f32 = 0.;           // X=0
            let mut y : f32 = - a / 25.;    // Y=-A/25:
            let mut z : f32 = a;           // Z=A
            let mut i : f32 = -1.;          // I=SGN(M-80.5)
            if mf >= 80.5 {i = 1.;}

            let mut u : f32 = (mf - 80.5) / (40. * 1.3);    // line 3; U=(M-80.5)/(40*1.3)
            let mut v : f32 = (nf - 80.5) / (80. * 1.3);    // V=(N-80.5)/(80*1.3)
            let mut w : f32 = 1. / f32::sqrt(u * u + v * v + 1.);  // W=1/SQR(U*U+V*V+1)
            u = u * w;                                    // U=U*W
            v = v * w;                                    // V=V*W
            let mut g : f32 = 1.;                         // G=1

            let mut p: f32 = 0.;
            loop {
                let mut e : f32 = x - i;                      // line 4; E=X-I
                let mut f : f32 = y - i;                      // F=Y-I
                p = u * e + v * f - w * z;                    // P=U*E+V*F-W*Z
                let d : f32 = p * p - e * e - f * f - z * z + 1.; // D=P*P-E*E-F*F-Z*Z+1

                // goto:  G.5+(D<=0)*3   // if d<=0, goto line 8 (break)
                if d <= 0. {break;}

                let mut t : f32 = -p - f32::sqrt(d);          // line 5; T=-P-SQR(D)
                // goto: G.6+(T<=0)*2    // if t<=0, goto line 8 (break)
                if t <= 0. {break;}

                x = x + t * u;  // line 6; X=X+T*U
                y = y + t * v;  // Y=Y+T*V
                z = z - t * w;  // Z=Z-T*W
                e = x - i;      // E=X-I
                f = y - i;      // F=Y-I
                g = z;          // G=Z
                p = 2. * (u * e + v * f - w * g);   // P=2*(U*E+V*F-W*G)

                u = u - p * e;  // line 7; U=U-P*E
                v = v - p * f;  // V=V-P*F
                w = w + p * g;  // W=W+P*G
                i = -i;         // I=-I

            }   // goto: G.4

            if v < 0. {p = (y + 2.) / v;}      // line 8; IF V<0THEN P=(Y+2)/V

            //let mut s : f32 = f32::floor(x - u * p) + f32::floor(z - w * p);  // S=INT(X-U*P)+INT(Z-W*P)
            let mut s : i32 = (f32::floor(x - u * p) + f32::floor(z - w * p)) as i32;  // S=INT(X-U*P)+INT(Z-W*P)

            // s = s - f32::floor(s / 2.) * 2.;   // S=S-INT(S/2)*2
            s = s % 2; // s - f32::floor(s / 2.) * 2.;   // S=S-INT(S/2)*2    // Chessboard

            v = -v * ((s as f32) / 2. + 0.3) + 0.2;    // V=-V*(S/2+.3)+.2

            // line 9 C.3-INT(((3*16)*SQR(V)+DI((M-INT(M/4)*4)+(N-INT(N/4)*4)*4)/3)/16)
            //let m_diff : usize = usize::try_from(m - ((m / 4)) * 4).unwrap();
            //let m_mod : usize = usize::try_from(m % 4).unwrap();
            //let m_mod : usize = usize::try_from(m_fact % 4).unwrap();
            //let n_diff : usize = usize::try_from(n - ((n / 4)) * 4).unwrap();
            //let n_mod : usize = usize::try_from(n % 4).unwrap();
            //let n_mod : usize = usize::try_from(n_fact % 4).unwrap();

            //let di_index : usize = m_mod + n_mod * 4;
            // Pixel color
            let mut c: u8 = (255. * v) as u8;

            // PL.M,191-N - system instruction: PLOT(M, 191 - N)
            c = 255 - c;

            if c < 50 {
                *pixel = Rgba([c, 0, c, 255]);
            }
            else if c < 100 {
                *pixel = Rgba([50, c - 50, 50, 255]);
            } 
            else if c < 150 {
                *pixel = Rgba([50 + (c - 100), 50 + (c - 100), 100, 255]);
            }
            else {
                *pixel = Rgba([100, 100 + (c - 150), 100 + (c - 100), 255]);
            }

            //} // for m in 0..159 {            // line 10; N.M
        //} // for n in 0..191 {            // N.N  (next n)
        } // for (n, m, pixel) in image.enumerate_pixels_mut() {

        // A=10*(A=-1)+(A-.1)*(A>-1)
        /*
        if a <= -1.
            {a = 10.;}
        else
            {a = a - 0.1;}
        */
     
        if (img_file_prefix.len() > 1) {
            let img_f_name : String = format!("{}{:0>3}.png", img_file_prefix, img_index);
            let res : ImageResult<()> = image.save(&img_f_name);
            println!("Saved image : {}", &img_f_name);
        }
    }   // goto: G.2      // for img_index in 0..n_img_iter {

} // fn raytracing_single_thread()


// Atari -> Rust port (CPU) with higher resolution, higher number of imgs/sec, improved graphics, multithreading
pub fn raytracing_cpu_multithreading(n_img_iter: i32, factor_res: i32, n_threads: u32, img_file_prefix: &String)
{

    let n_img_per_thread: u32 = ((n_img_iter as f32) / (n_threads as f32)).ceil() as u32;

    static GLOBAL_THREAD_COUNT: AtomicUsize = ATOMIC_USIZE_INIT;
    
    for thread_index in 0..n_threads
    {
        let img_index_min: i32 = (n_img_per_thread * thread_index) as i32;
        let mut img_index_max: i32 = (n_img_per_thread * thread_index + n_img_per_thread) as i32;
        if img_index_max >= n_img_iter {img_index_max = n_img_iter - 1;}
        let img_file_prefix_clone: String = img_file_prefix.clone();

        GLOBAL_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
        let handle = std::thread::spawn( move ||
        {
            raytracing_cpu_single_thread(&img_index_min, &img_index_max, &n_img_iter, &factor_res, &img_file_prefix_clone);
            GLOBAL_THREAD_COUNT.fetch_sub(1, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(1));
            //println!("Thread has terminated {} / {}", thread_index, n_threads);
        });
        //handle.join();
    }
    
    //let n_sec:  u64 = 120;
    //let n_msec: u64 = n_sec * 1000;
    //std::thread::sleep(std::time::Duration::from_millis(n_msec));
    while GLOBAL_THREAD_COUNT.load(Ordering::SeqCst) != 0 {
        thread::sleep(Duration::from_millis(1)); 
    }

} // pub fn raytracing_cpu_multithreading(n_img_iter: i32, factor_res: i32, n_threads: u32, img_file_prefix: &String)



// Input: A "1D Tensor" of size B, where B is the batchsize.
// It contains the only information which is required : The image index
fn raytracing_gpu_tch_forward(t_input: Tensor, ref_batch_size: &i32, ref_n_img_iter : &i32, ref_factor_res: &i32, n_max_reflections: u32) -> Tensor
{
    // Internal flag for debugging
    let b_fixed_one_reflection: bool = false;

    let mut vs = VarStore::new(Device::cuda_if_available());

    let t_img_index: Tensor = t_input.to_device(vs.device());

    let batch_size: i64    = *ref_batch_size as i64;
    let n_img_iter: i32        = *ref_n_img_iter;
    let factor_res: i32    = *ref_factor_res;

    // line 1
    // GR.31: Set resolution 160x192 (system instruction)
    let cx: i32 = 160;
    let cy: i32 = 192;

    // Atari dithering not used here
    // DI.DI(16): F.I=0TO15: REA.R: DI(I)=R: N.I

    //A=2
    // A=10*(A=-1)+(A-.1)*(A>-1) (code line 10 on atari BASIC)
    //let mut a : f32 = 2. - (img_index as f32) * 0.1 / (n_img_iter as f32) * 110.;  // rust CPU
    let mut t_a_small: Tensor = 2. - t_img_index * (0.1 / (n_img_iter as f64) * 110.);

    //if a <= -1. {a = a + 11.;}
    t_a_small = &t_a_small + Tensor::where_scalar(&t_a_small.less_equal(-1.), 11., 0.);
    // see https://pytorch.org/docs/stable/generated/torch.where.html
    // and https://docs.rs/tch/latest/tch/struct.Tensor.html#method.where_scalar
    // and https://docs.rs/tch/latest/tch/struct.Tensor.html#method.less_equal



    //let mut image : RgbaImage = RgbaImage::new((cx * 2 * factor_res) as u32, (cy * factor_res) as u32);
    let height: i64 = (cy * factor_res) as i64;
    let width: i64 = (cx * 2 * factor_res) as i64;  // Factor 2 because of atari pixels size


    // Same value for 'a', for all the pixels.
    // https://pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch.unsqueeze
    // https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand
    // https://docs.rs/tch/0.15.0/tch/struct.Tensor.html#method.unsqueeze
    // https://docs.rs/tch/0.15.0/tch/struct.Tensor.html#method.expand
    // In pytorch: A.unsqueeze(1).unsqueeze(2).expand(B, H, W)
    //let int_list_vec : Vec<i64> = vec![batch_size as i64, height as i64, width as i64];
    //let int_list: dyn IntList = int_list_vec as dyn IntList;
    //let t_a: Tensor = t_a_small.unsqueeze(1).unsqueeze(2).expand(int_list, false);
    let t_a: Tensor = t_a_small.unsqueeze(1).unsqueeze(2).expand([batch_size, height, width], false);
       

    //        for n in 0..191 {              // line 2;  F.N=0TO191- loop over lines
    let n_indexes: Vec<i64> = (0..height).collect::<Vec<_>>();
    // Same value for 'n', for all the images, and all pixels of same line
    let t_im_y_fact: Tensor = Tensor::from_slice(&n_indexes).unsqueeze(0).unsqueeze(2).expand([batch_size, height, width], false).to_device(vs.device());

    //            for m in 0..159 {            // F.M=0TO159     // Loop over pixels of current line
    let m_indexes: Vec<i64> = (0..width).collect::<Vec<_>>();
    // Same value for 'n', for all the images, and all pixels of same line
    let t_im_x_fact: Tensor = Tensor::from_slice(&m_indexes).unsqueeze(0).unsqueeze(1).expand([batch_size, height, width], false).to_device(vs.device());

    //  for (im_x_fact, im_y_fact, pixel) in image.enumerate_pixels_mut() {    // pc CPU (previous) version of loop over pixels of an image

    // Here, we are inside 3 'virtual' loops: Looping over images of the batch, over lines of the same image, and over pixels of the same line.
    // -> plain GPU-parrallele computing
    // Torch style Tensors: B * H * W ; BatchSize * Height * Weight.
    // We skipped C, the channel, equals to 1 (Variables treated separately, i.e. 1 Tensor equals 1 atari BASIC variable).

    let t_m_fact: Tensor = 0.5 * t_im_x_fact;
    let t_n_fact: Tensor = 1.  * (192 * factor_res - t_im_y_fact);

    let t_mf: Tensor = t_m_fact / (factor_res as f64);
    let t_nf: Tensor = t_n_fact / (factor_res as f64);
    
    //let t_m: Tensor = 1. * &t_mf;
    //let t_n: Tensor = 1. * &t_nf;

    //let mut x : f32 = 0.;           // X=0
    // use repeat and not expand, as the values of the array will change, and have their own life.
    let mut t_x : Tensor = Tensor::from(0f64).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat([batch_size, height, width]).to_device(vs.device());

    //let mut y : f32 = - a / 25.;    // Y=-A/25:
    let mut t_y : Tensor = -1. * &t_a / 25.;
    //let mut z : f32 = a;           // Z=A
    let mut t_z : Tensor = 1. * &t_a;

    //let mut i : f32 = -1.;          // I=SGN(M-80.5)
    let mut t_i : Tensor = Tensor::from(-1f64).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat([batch_size, height, width]).to_device(vs.device());
    //if mf >= 80.5 {i = 1.;}
    t_i =  t_i.where_scalarother(&t_mf.less(80.5), 1.);    // Certainement correct
    //t_i =  t_i.where_scalarother(&t_mf.greater_equal(80.5), 1.);

    //let mut u : f32 = (mf - 80.5) / (40. * 1.3);    // line 3; U=(M-80.5)/(40*1.3)
    let mut t_u : Tensor = (t_mf - 80.5) / (40. * 1.3);

    //let mut v : f32 = (nf - 80.5) / (80. * 1.3);    // V=(N-80.5)/(80*1.3)
    let mut t_v : Tensor = (t_nf - 80.5) / (80. * 1.3);

    //let mut w : f32 = 1. / f32::sqrt(u * u + v * v + 1.);  // W=1/SQR(U*U+V*V+1)
    let t_w_tmp : Tensor = &t_u * &t_u + &t_v * &t_v + 1.;
    let mut t_w : Tensor = 1. / Tensor::sqrt(&t_w_tmp);

    //u = u * w;                                    // U=U*W
    t_u = &t_u * &t_w;

    //v = v * w;                                    // V=V*W
    t_v = &t_v * &t_w;

    //let mut g : f32 = 1.;                         // G=1
    let mut t_g : Tensor = Tensor::from(1f64).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat([batch_size, height, width]).to_device(vs.device());

    //let mut p: f32 = 0.;
    let mut t_p : Tensor = Tensor::from(0f64).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat([batch_size, height, width]).to_device(vs.device());

    let mut t_e: Tensor = Tensor::from(0f64).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat([batch_size, height, width]).to_device(vs.device());
    let mut t_f: Tensor = Tensor::from(0f64).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat([batch_size, height, width]).to_device(vs.device());
    let mut t_d: Tensor = Tensor::from(0f64).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat([batch_size, height, width]).to_device(vs.device());
    let mut t_t: Tensor = Tensor::from(0f64).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat([batch_size, height, width]).to_device(vs.device());


    if (b_fixed_one_reflection)
    {
        // Force one fixed reflection for all pixels. Used for debugging vs the CPU-version,
        // before implementing the variable reflections number (among the pixels).

        //let mut e : f32 = x - i;                      // line 4; E=X-I
        t_e = &t_x - &t_i;

        //let mut f : f32 = y - i;                      // F=Y-I
        t_f = &t_y - &t_i;

        //p = u * e + v * f - w * z;                    // P=U*E+V*F-W*Z
        t_p = &t_u * &t_e + &t_v * &t_f - &t_w * &t_z;

        //let d : f32 = p * p - e * e - f * f - z * z + 1.; // D=P*P-E*E-F*F-Z*Z+1
        t_d = &t_p * &t_p - &t_e * &t_e - &t_f * &t_f - &t_z * &t_z + 1.;

        // goto:  G.5+(D<=0)*3   // if d<=0, goto line 8 (break)
        //if d <= 0. {break;}
        // All pixels bounce once

        //let mut t : f32 = -p - f32::sqrt(d);          // line 5; T=-P-SQR(D)
        t_t = -&t_p - Tensor::sqrt(&t_d);

        // goto: G.6+(T<=0)*2    // if t<=0, goto line 8 (break)
        //if t <= 0. {break;}
        // All pixels bounce once

        //x = x + t * u;  // line 6; X=X+T*U
        t_x = &t_x + &t_t * &t_u;

        //y = y + t * v;  // Y=Y+T*V
        t_y = &t_y + &t_t * &t_v;

        //z = z - t * w;  // Z=Z-T*W
        t_z = &t_z - &t_t * &t_w;

        //e = x - i;      // E=X-I
        t_e = &t_x - &t_i;

        //f = y - i;      // F=Y-I
        t_f = &t_y - &t_i;

        //g = z;          // G=Z
        t_g = 1. * &t_z;

        //p = 2. * (u * e + v * f - w * g);   // P=2*(U*E+V*F-W*G)
        t_p = 2. * (&t_u * &t_e + &t_v * &t_f - &t_w * &t_g);

        //u = u - p * e;  // line 7; U=U-P*E
        t_u = &t_u - &t_p * &t_e;

        //v = v - p * f;  // V=V-P*F
        t_v = &t_v - &t_p * &t_f;

        //w = w + p * g;  // W=W+P*G
        t_w = &t_w + &t_p * &t_g;

        //i = -i;         // i=-i
        t_i = -1. * &t_i;
    }
    else {

        // Loop to calculate the ray reflections on spheres
        // The number of reflections among the pixels may vary
        // Parrallele computing prevent pixel-level plain rust instructions
        // So we need to use a boolean tensor to keep track of alive rays during the loop iterations
        // and perform computation only on pixels which did not 'loop break'.
        // Mind that this is rather unefficient, as most of the pixels will have ray bouncing 1 or 2 times maximum,
        // but some rays bound up to 8 times, according to stats from the CPU implementation.
        // On the other side, playing with tensors at this level and dynamically extract alive rays only,
        // as a way to treat those alive only, may lead to performance drop as well
        // (because of mem allocs bottleneck and/or break parrallele computing high flow).

        // Feed with '1' values: All alives at the beginning.
        let mut t_alive:Tensor = Tensor::from(1f64).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat([batch_size, height, width]).to_device(vs.device());
        let mut t_other:Tensor = Tensor::new();
        let mut count_iter : u32 = 0;
        loop {

            // Loop while at least one CUDA kernel is alive, i.e. rays are bouncing between spheres
            
            //let mut e : f32 = x - i;                      // line 4; E=X-I
            //t_e = &t_x - &t_i;
            t_other = &t_x - &t_i;
            t_e = t_e.where_self(&t_alive.less(0.5), &t_other);

            //let mut f : f32 = y - i;                      // F=Y-I
            //t_f = &t_y - &t_i;
            t_other = &t_y - &t_i;
            t_f = t_f.where_self(&t_alive.less(0.5), &t_other);

            //p = u * e + v * f - w * z;                    // P=U*E+V*F-W*Z
            //t_p = &t_u * &t_e + &t_v * &t_f - &t_w * &t_z;
            t_other = &t_u * &t_e + &t_v * &t_f - &t_w * &t_z;
            t_p = t_p.where_self(&t_alive.less(0.5), &t_other);

            //let d : f32 = p * p - e * e - f * f - z * z + 1.; // D=P*P-E*E-F*F-Z*Z+1
            //t_d = &t_p * &t_p - &t_e * &t_e - &t_f * &t_f - &t_z * &t_z + 1.;
            t_other = &t_p * &t_p - &t_e * &t_e - &t_f * &t_f - &t_z * &t_z + 1.;
            t_d = t_d.where_self(&t_alive.less(0.5), &t_other);

            // goto:  G.5+(D<=0)*3   // if d<=0, goto line 8 (break)
            //if d <= 0. {break;}
            // Update of the alive tensor with rays that won't bound anymore.
            t_alive = t_alive * Tensor::where_scalar(&t_d.less_equal(0.), 0., 1.);

            //let mut t : f32 = -p - f32::sqrt(d);          // line 5; T=-P-SQR(D)
            //t_t = -&t_p - Tensor::sqrt(&t_d);
            t_other = -&t_p - Tensor::sqrt(&t_d);
            t_t = t_t.where_self(&t_alive.less(0.5), &t_other);

            // goto: G.6+(T<=0)*2    // if t<=0, goto line 8 (break)
            //if t <= 0. {break;}
            // Update of the alive tensor with rays that won't bound anymore.
            t_alive = t_alive * Tensor::where_scalar(&t_t.less_equal(0.), 0., 1.);

            //x = x + t * u;  // line 6; X=X+T*U
            //t_x = &t_x + &t_t * &t_u;
            t_other = &t_x + &t_t * &t_u;
            t_x = t_x.where_self(&t_alive.less(0.5), &t_other);

            //y = y + t * v;  // Y=Y+T*V
            //t_y = &t_y + &t_t * &t_v;
            t_other = &t_y + &t_t * &t_v;
            t_y = t_y.where_self(&t_alive.less(0.5), &t_other);

            //z = z - t * w;  // Z=Z-T*W
            //t_z = &t_z - &t_t * &t_w;
            t_other = &t_z - &t_t * &t_w;
            t_z = t_z.where_self(&t_alive.less(0.5), &t_other);

            //e = x - i;      // E=X-I
            //t_e = &t_x - &t_i;
            t_other = &t_x - &t_i;
            t_e = t_e.where_self(&t_alive.less(0.5), &t_other);

            //f = y - i;      // F=Y-I
            //t_f = &t_y - &t_i;
            t_other = &t_y - &t_i;
            t_f = t_f.where_self(&t_alive.less(0.5), &t_other);

            //g = z;          // G=Z
            //t_g = 1. * &t_z;
            t_other = 1. * &t_z;
            t_g = t_g.where_self(&t_alive.less(0.5), &t_other);

            //p = 2. * (u * e + v * f - w * g);   // P=2*(U*E+V*F-W*G)
            //t_p = 2. * (&t_u * &t_e + &t_v * &t_f - &t_w * &t_g);
            t_other = 2. * (&t_u * &t_e + &t_v * &t_f - &t_w * &t_g);
            t_p = t_p.where_self(&t_alive.less(0.5), &t_other);

            //u = u - p * e;  // line 7; U=U-P*E
            //t_u = &t_u - &t_p * &t_e;
            t_other = &t_u - &t_p * &t_e;
            t_u = t_u.where_self(&t_alive.less(0.5), &t_other);

            //v = v - p * f;  // V=V-P*F
            //t_v = &t_v - &t_p * &t_f;
            t_other = &t_v - &t_p * &t_f;
            t_v = t_v.where_self(&t_alive.less(0.5), &t_other);

            //w = w + p * g;  // W=W+P*G
            //t_w = &t_w + &t_p * &t_g;
            t_other = &t_w + &t_p * &t_g;
            t_w = t_w.where_self(&t_alive.less(0.5), &t_other);

            //i = -i;         // i=-i
            //t_i = -1. * &t_i;
            t_other = -1. * &t_i;
            t_i = t_i.where_self(&t_alive.less(0.5), &t_other);

            count_iter = count_iter + 1;

            //println!("iter cour {}", count_iter);
            
            // If all rays are finished bouncing, break the loop for everybody
            //let n_alive = t_alive.sum(tch::kind::Kind::Float).double_value(&[0 as i64]);
            //if n_alive < 0.5 {break;}

            // Hardcoded max number of reflections for the moment (Need to fix the commented lines above)
            // (Anyway, multithreading version debugging shows there are less than 10 rebounds, difference hardly seen with eyes above 4 ou 5).
            if (count_iter >= n_max_reflections) {break;}
        } // end loop
        //println!("Nb iter max {}", count_iter);

    } // end if/else


    //if v < 0. {p = (y + 2.) / v;}      // line 8; if v<0then p=(y+2)/v
    let t_p_tmp = (&t_y + 2.) / &t_v;
    t_p = t_p.where_self(&t_v.greater_equal(0.), &t_p_tmp);
    //t_p = t_p.where_self(&t_v.less(0.), &t_p_tmp);

    //let mut s : i32 = (f32::floor(x - u * p) + f32::floor(z - w * p)) as i32;  // s=int(x-u*p)+int(z-w*p)
    let t_s_tmp1: Tensor = &t_x - &t_u * &t_p;
    let t_s_tmp2: Tensor = &t_z - &t_w * &t_p;
    let mut t_s: Tensor = Tensor::floor(&t_s_tmp1) + Tensor::floor(&t_s_tmp2);
    //t_s = t_s.to_kind(tch::kind::Kind::Int64);

    //s = s % 2; // s - f32::floor(s / 2.) * 2.;   // s=s-int(s/2)*2    // chessboard
    let t_s_tmp3 = &t_s / 2.;
    t_s = &t_s - Tensor::floor(&t_s_tmp3) * 2.;
    //t_s = t_s % 2;
    //t_s = t_s.to_kind(tch::kind::Kind::Double);

    //v = -v * ((s as f32) / 2. + 0.3) + 0.2;    // v=-v*(s/2+.3)+.2
    t_v = -1. * &t_v * (&t_s / 2. + 0.3) + 0.2;

    // pl.m,191-n - system instruction: plot(m, 191 - n)
    // pixel color 'intensity'
    let mut t_c: Tensor = 1. - &t_v;

    // Add some color interpolation for display
    let rgb_threshold: Vec<f64> = vec![0.4, 0.4, 0.4];
    let rgb_left: Vec<f64>      = vec![1.4, 0.6, 0.0];
    let rgb_mid: Vec<f64>       = vec![0.6, 0.4, 0.0];
    let rgb_right: Vec<f64>     = vec![-0.4, -0.5, 0.0];

    let t_pixel_red_left    : Tensor = rgb_left[0] + ((rgb_mid[0]   - rgb_left[0]) / rgb_threshold[0]) * &t_c;
    let t_pixel_green_left  : Tensor = rgb_left[1] + ((rgb_mid[1]   - rgb_left[1]) / rgb_threshold[1]) * &t_c;
    let t_pixel_blue_left   : Tensor = rgb_left[2] + ((rgb_mid[2]   - rgb_left[2]) / rgb_threshold[2]) * &t_c;

    let t_pixel_red_right   : Tensor = rgb_mid[0]  + ((rgb_right[0] - rgb_mid[0])  / (1. - rgb_threshold[0])) * (&t_c - rgb_threshold[0]);
    let t_pixel_green_right : Tensor = rgb_mid[1]  + ((rgb_right[1] - rgb_mid[1])  / (1. - rgb_threshold[1])) * (&t_c - rgb_threshold[1]);
    let t_pixel_blue_right  : Tensor = rgb_mid[2]  + ((rgb_right[2] - rgb_mid[2])  / (1. - rgb_threshold[2])) * (&t_c - rgb_threshold[2]);

    let t_pixel_red   : Tensor =  t_pixel_red_left.where_self(&t_c.less(rgb_threshold[0]), &t_pixel_red_right).unsqueeze(1);
    let t_pixel_green : Tensor =  t_pixel_green_left.where_self(&t_c.less(rgb_threshold[1]), &t_pixel_green_right).unsqueeze(1);
    let t_pixel_blue  : Tensor =  t_pixel_blue_left.where_self(&t_c.less(rgb_threshold[2]), &t_pixel_blue_right).unsqueeze(1);

    // Eventually get a pytorch-like tensor shape [B, C, H, W]
    let t_pixel_rgb : Tensor = Tensor::cat(&[t_pixel_red, t_pixel_green, t_pixel_blue], 1);

    let vs_cpu = tch::nn::VarStore::new(Device::Cpu);
    return t_pixel_rgb.to_device(vs_cpu.device());

} // fn raytracing_gpu_tch_forward(t_input: Tensor, ref_batch_size: &i32, ref_n_img_iter : &i32, ref_factor_res: &i32, b_fixed_one_reflection: bool) -> Tensor


fn raytracing_gpu_tch(n_img_iter: i32, factor_res: i32, batch_size: i32, n_output_threads: u32, b_use_autocast: bool, n_max_reflections: u32, img_file_prefix: &String)
{
    let n_batch: u32 = ((n_img_iter as f32) / (batch_size as f32)).ceil() as u32;

    for batch_index in 0..n_batch
    {
        let img_index_min: i32 = batch_size * (batch_index as i32);
        let mut img_index_max: i32 = img_index_min + (batch_size as i32);
        if img_index_max > n_img_iter {img_index_max = n_img_iter;}
        let effective_batch_size: i32 = img_index_max - img_index_min;

        let img_indexes: Vec<i32> = (img_index_min..img_index_max).collect::<Vec<_>>();
        let t_input: Tensor = Tensor::from_slice(&img_indexes);

        let mut t_pixels: Tensor = tch::Tensor::new();
        // https://docs.rs/tch/0.15.0/tch/fn.no_grad.html
        // https://pytorch.org/docs/stable/generated/torch.no_grad.html
        no_grad( || {
            // https://docs.rs/tch/0.15.0/tch/fn.autocast.html
            // https://pytorch.org/docs/stable/amp.html#torch.autocast
            autocast( b_use_autocast,  || {
                t_pixels = raytracing_gpu_tch_forward(t_input, &effective_batch_size, &n_img_iter, &factor_res, n_max_reflections);
            });
        });

        let cx: i32 = 160;
        let cy: i32 = 192;
        let height = (cy * factor_res) as i32;
        let width = (cx * 2 * factor_res) as i32;

        //let t_pixels : Tensor = t_output.unsqueeze(1).expand([effective_batch_size as i64, 3 as i64, height as i64, width as i64], false);

        // (CPU) Multithreading of the jpg compression for output images
        let n_img_per_thread: i32 = (((img_index_max - img_index_min) as f32) / (n_output_threads as f32)).ceil() as i32;
        static GLOBAL_THREAD_COUNT: AtomicUsize = ATOMIC_USIZE_INIT;
        
        // We split this loop across the threads: 'for img_index in img_index_min..img_index_max'
        for thread_index in 0..n_output_threads
        {
            let img_index_min_thread: i32 = (img_index_min + n_img_per_thread * (thread_index as i32)) as i32;
            let mut img_index_max_thread: i32 = (img_index_min_thread + n_img_per_thread) as i32;
            if img_index_max_thread > img_index_max {img_index_max_thread = img_index_max;}
            let img_file_prefix_clone: String = img_file_prefix.clone();

            // https://docs.rs/tch/0.15.0/tch/struct.Tensor.html#method.shallow_clone
            let t_pixels_thread: Tensor = t_pixels.shallow_clone();  // Not a real copy = no additional used memory.

            GLOBAL_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
            let handle = std::thread::spawn( move ||
            {
                let img_index = 0;
                for img_index in img_index_min_thread..img_index_max_thread
                {
                    let batch_img_index: u32 = (img_index - img_index_min) as u32;
                    let img_f_name : String = format!("{}{:0>3}.jpg", img_file_prefix_clone, img_index);
                    let Result = save_image(&t_pixels_thread.get(batch_img_index as i64), &img_f_name);
                    println!("Saved image : {}", &img_f_name);
                }
                GLOBAL_THREAD_COUNT.fetch_sub(1, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(1));
            });
        }
        while GLOBAL_THREAD_COUNT.load(Ordering::SeqCst) != 0 {
            thread::sleep(Duration::from_millis(1)); 
        }
    }
} // fn raytracing_gpu_tch(n_img_iter: i32, factor_res: i32, batch_size: i32, n_output_threads: u32, b_use_autocast: bool, b_fixed_one_reflection: bool)



fn main() {

    if (true) {
        // Plain translation from Atari BASIC to Rust
        let factor_res: u32 = 1;   // Original resolution 160x192 (without doubling the horizontal pixels)
        let img_file_prefix: String = String::from("generated_imgs/img_atari_");
        raytracing_cpu_atari_like(factor_res, &img_file_prefix);
    }

    if (true) {
        // Translation to Rust + img quality improvements + multithreading
        let n_img_iter: i32 = 300; // Increase the images rate (x3)
        let factor_res: i32 = 5;   // Increase the resolution to near HD
        let n_threads: u32 = 16;   // Number of threads (should match the nb of cpu core)
        let img_file_prefix: String = String::from("generated_imgs/img_cpu_");
        raytracing_cpu_multithreading(n_img_iter, factor_res, n_threads, &img_file_prefix);
    }

    if (true) {
        // Rust + gpu/cuda through the tch-rs library (wrapper for libtorch)
        let n_img_iter: i32 = 600; // Increase the images rate (x6)
        let factor_res: i32 = 5;   // Increase the resolution to near HD
        let batch_size: i32 = 18;  // Number of images treated simultaneously on GPU (Need to be adjusted depending on your VRAM)
        let n_output_threads:  u32 = 16;   // Number of threads for jpeg compression when saving imgs
        let b_use_autocast: bool = true; // GPU computation in half precision (should save both memory and time)
        let n_max_reflections: u32 = 4;  // Limit the max number of iterations for reflections computation
        let img_file_prefix: String = String::from("generated_imgs/img_gpu_");
        raytracing_gpu_tch(n_img_iter, factor_res, batch_size, n_output_threads, b_use_autocast, n_max_reflections, &img_file_prefix);
    }

}