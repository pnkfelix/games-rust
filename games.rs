fn main() {
    games::antichess();
}


pub mod games {

    pub struct offset_range {
        min: char,
        max: char,
    }

    struct offset {
        base: char,
        magnitude: uint,
    }

    impl offset {
        fn magnitude(&self) -> uint { self.magnitude }
    }

    pub fn offset_range(min: char, max: char) -> offset_range {
        offset_range{ min: min, max: max }
    }

    impl offset_range {
        fn inject(&self, c: char) -> Option<offset> {
            if self.min <= c && c <= self.max {
                Some(offset {
                        base: self.min,
                        magnitude: c as uint - self.min as uint
                    })
            } else {
                None
            }
        }
    }


    pub mod chess {
        use super::offset_range;

        #[deriving(ToStr,Clone,Eq)]
        pub enum Color { black, white }

        impl Color {
            fn rev(self) -> Color {
                match self { black => white, white => black }
            }
        }

        #[deriving(Clone)]
        pub enum Piece { king, queen, rook, bishop, knight, pawn }

        condition! {
            pub malformed_piece : char -> super::Piece;
        }

        impl Piece {
            pub fn from_char(c: char) -> Piece {
                match c {
                    'R' | 'r' => rook,
                    'N' | 'n' | 'S' | 's' => knight,
                    'B' | 'b' | 'F' | 'f' => bishop,
                    'Q' | 'q' => queen,
                    'K' | 'k' => king,
                    'P' | 'p' => pawn,
                    _ => malformed_piece::cond.raise(c)
                }
            }
        }

        #[deriving(Clone)]
        pub struct Man(Color, Piece);

        impl Man {
            pub fn to_str(self) -> ~str {
                let Man(c,p) = self;
                match (c,p) {
                    (white,king)   => ~"\u2654",
                    (white,queen)  => ~"\u2655",
                    (white,rook)   => ~"\u2656",
                    (white,bishop) => ~"\u2657",
                    (white,knight) => ~"\u2658",
                    (white,pawn)   => ~"\u2659",

                    (black,king)   => ~"\u265A",
                    (black,queen)  => ~"\u265B",
                    (black,rook)   => ~"\u265C",
                    (black,bishop) => ~"\u265D",
                    (black,knight) => ~"\u265E",
                    (black,pawn)   => ~"\u265F",
                }
            }
        }

        struct Board {
            rows: [[Option<Man>, ..8], ..8]
        }

        impl Clone for Board {
            fn clone(&self) -> Board {
                Board{ rows: self.rows }
            }
        }

        #[deriving(Eq)]
        pub struct Row(uint);

        #[deriving(Eq)]
        pub struct Col(uint);

        impl Row {
            fn maybe(mag:Option<uint>) -> Option<Row> {mag.map(|m| { Row(*m) })}
            pub fn from_char(c:char) -> Option<Row> {
                Row::maybe(offset_range('1', '8').inject(c).map(|x|x.magnitude()))
            }
            fn magnitude(&self) -> uint { **self }
        }

        impl Col {
            fn maybe(mag:Option<uint>) -> Option<Col> {mag.map(|m| { Col(*m) })}
            pub fn from_char(c:char) -> Option<Col> {
                Col::maybe(offset_range('a', 'h').inject(c).map(|x|x.magnitude()))
            }
            fn magnitude(&self) -> uint { **self }
        }

        #[deriving(Eq)]
        pub struct Square { letter: Col, number: Row }

        impl ToStr for Square {
            fn to_str(&self) -> ~str {
                fmt!("%c%c",
                     ('a' as uint + *self.letter) as char,
                     ('1' as uint + *self.number) as char)
            }
        }

        impl Board {
            pub fn to_str(&self) -> ~str {
                let mut accum = ~" abcdefgh\n";
                let mut count = 8;
                for row in self.rows.rev_iter() {
                    accum = accum + fmt!("%u", count);
                    let mut square_color =
                        if count % 2 == 0 { white } else { black };
                    for cell in row.iter() {
                        accum = accum + match *cell {
                            None => match square_color {
                                black => ~"\u25a0",
                                white => ~"\u25a1",
                            },
                            Some(m) => m.to_str()
                        };
                        square_color = square_color.rev();
                    }
                    accum = accum + fmt!("%u", count) + "\n";
                    count -= 1;
                }
                accum = accum + " abcdefgh";
                fmt!("%s", accum)
            }

            pub fn cell<'a>(&'a self, s: Square) -> &'a Option<Man> {
                &self.rows[s.number.magnitude()][s.letter.magnitude()]
            }

            pub fn cell_mut<'a>(&'a mut self, s: Square) -> &'a mut Option<Man> {
                &mut self.rows[s.number.magnitude()][s.letter.magnitude()]
            }

            pub fn at(&self, s: Square) -> Option<Man> { self.cell(s).clone() }
            pub fn put(&mut self, s: Square, m: Man) {*self.cell_mut(s) = Some(m);}
            pub fn clear(&mut self, s: Square) {*self.cell_mut(s) = None;}
        }

        pub type Move = (Square, Square);

        #[deriving(ToStr)]
        pub enum InvalidMoveReason {
            source_and_target_are_same_square(Square),
            source_square_is_empty(Square),
            piece_is_not_your_color(Man),
            target_square_holds_piece_of_your_color(Man),
        }

        impl InvalidMoveReason {
            // good enough for now
            pub fn reason(self) -> ~str { self.to_str() }
        }

        // Can respond to an invalid move by providing a different one
        // to apply.
        condition! {
            pub invalid_move : (super::InvalidMoveReason,
                                super::Move,
                                super::Game) -> super::Move;
        }

        #[deriving(Clone)]
        struct Game {
            board: Board,
            current: Color,
            black_taken: ~[Piece],
            white_taken: ~[Piece]
        }

        pub fn pieces_to_str(c: Color, v: &[Piece]) -> ~str {
            v.map(|p| Man(c,*p).to_str()).concat()
        }

        impl Game {
            pub fn to_str(&self) -> ~str {
                let ret = self.board.to_str();
                let ret = ret.replace("8\n",
                                      fmt!("8 %s \n", pieces_to_str(black, self.black_taken)));
                let ret = ret.replace("1\n",
                                      fmt!("1 %s \n", pieces_to_str(white, self.white_taken)));
                let ret = ret + "\n" + self.current.to_str() + "'s move";
                ret
            }

            pub fn validate_move(&mut self, move: Move) -> (Move, Man, Option<Man>) {
                let mut move = move;

                let raise = |reason| {
                    invalid_move::cond.raise((reason, move, self.clone()))
                };

                macro_rules! retry(
                    ($reason:expr) => { move = raise($reason); loop; }
                );

                macro_rules! retry2(
                    ($d:ident, $reason:expr) => { $d = raise($reason); loop; }
                );

                loop {
                    let (from, to) = move;

                    if from == to {
                        let r = source_and_target_are_same_square(from);
                        // Neither of the below macros work.  It is not clear why.
                        // (I can expose via the input sequence: "c2 c2", "c2 c3")
                        // retry!(move, r)
                        // retry2!(move, r)
                        move = raise(r); loop;
                    }

                    let m = *self.board.cell(from);

                    debug!("validate move %? : man %?", move.to_str(), m);

                    match m {
                        None => {
                            move = raise(source_square_is_empty(from)); loop;
                        }
                        Some(source_man @ Man(color, _)) => {
                            if self.current != color {
                                move = raise(piece_is_not_your_color(source_man)); loop;
                            }

                            debug!("validate move %? : source %? to %?",
                                   move.to_str(), source_man, to.to_str());
                            let target_man : Option<Man> =
                                match *self.board.cell(to) {
                                // empty target is a-okay
                                None => {
                                    None
                                },
                                Some(man @ Man(color, _)) => {
                                    debug!("validate move %? : target man %?", move.to_str(), man);
                                    if self.current == color {
                                        move = raise(target_square_holds_piece_of_your_color(man));
                                        loop;
                                    } else {
                                        Some(man)
                                    }
                                },
                            };

                            // Okay. now everything has been checked.
                            return (move, source_man, target_man);
                        }
                    }
                    fail!("should never get here"); // is there a static_fail?
                }
            }

            pub fn do_move(&mut self, (from, to): Move) {
                let mut move = (from, to);

                let (move, source_man, target_man) = self.validate_move(move);
                let (from, to) = move;

                self.board.clear(from);
                self.board.put(to, source_man);

                match target_man {
                    Some(Man(black, p)) => self.black_taken.push(p),
                    Some(Man(white, p)) => self.white_taken.push(p),
                    None => {},
                }

                self.current = match self.current {
                    white => black,
                    black => white
                };
            }
        }

        static wp : Man = Man(white, pawn);
        static wr : Man = Man(white, rook);
        static wn : Man = Man(white, knight);
        static wb : Man = Man(white, bishop);
        static wq : Man = Man(white, queen);
        static wk : Man = Man(white, king);

        static bp : Man = Man(black, pawn);
        static br : Man = Man(black, rook);
        static bn : Man = Man(black, knight);
        static bb : Man = Man(black, bishop);
        static bq : Man = Man(black, queen);
        static bk : Man = Man(black, king);

        pub static initial_board : Board = Board {
            rows:
                [[Some(wr), Some(wn), Some(wb), Some(wq), Some(wk), Some(wb), Some(wk), Some(wr)],
                 [Some(wp), Some(wp), Some(wp), Some(wp), Some(wp), Some(wp), Some(wp), Some(wp)],
                 [None,     None,     None,      None,     None,    None,     None,     None    ],
                 [None,     None,     None,      None,     None,    None,     None,     None    ],
                 [None,     None,     None,      None,     None,    None,     None,     None    ],
                 [None,     None,     None,      None,     None,    None,     None,     None    ],
                 [Some(bp), Some(bp), Some(bp), Some(bp), Some(bp), Some(bp), Some(bp), Some(bp)],
                 [Some(br), Some(bn), Some(bb), Some(bq), Some(bk), Some(bb), Some(bk), Some(br)]],
        };
    }

    condition! {
        pub unreadable_move : ~str -> (super::chess::Square, super::chess::Square);
    }

    type ChessMove = chess::Move;

    pub fn get_move_recur(g: &chess::Game, inp: @Reader) -> ChessMove {
        print("? ");
        let input = inp.read_line();
        println("");
        debug!("Got input: %?",  input);
        do unreadable_move::cond.trap(|input| {
                println(fmt!("Could not parse input: <<%s>>", input));
                println("try again");
                print(fmt!("%s", g.to_str()));

                // XXX Rust is not tail-recursive, nor is this a
                // tail-recursive context.  So this pattern is bad.
                // However, (1.) we don't expect the user to make an
                // unbounded number of input errors, and (2.) if we
                // did expect that, we could keep a count and fail
                // after some attempt threshold.
                //
                // (The fix is probably to rewrite parse_move to
                // return Option instead of raising a condition; its
                // not like the code within parse uses the
                // continuability of the raised condition.)
                get_move_recur(g, inp)
            }).inside {
            parse_move(input.clone())
        }
    }

    pub fn get_move(g: &chess::Game, inp: @Reader) -> ChessMove {
        get_move_recur(g, inp)
    }

    pub fn read_square(input: &str) -> Option<(char, char, uint)> {
        let idx = input.find(|c| ('a' <= c && c <= 'h'));
        debug!("read_square idx: %?", idx);
        match idx {
            None => None,
            Some(idx) => {
                if idx+1 >= input.char_len() {
                    None
                } else {
                    let to_letter = input.char_at(idx);
                    let to_number = input.char_at(idx+1);
                    let val = (to_letter, to_number, idx+2);
                    debug!("read_square val: %?", val);
                    Some(val)
                }
            }
        }
    }

    pub fn parse_move(input: ~str) -> ChessMove {
        use games::chess::*;

        if input.char_len() < 2 {
            return unreadable_move::cond.raise(input.clone())
        }

        let (from_letter, from_number, post_idx) = match read_square(input) {
            None => return unreadable_move::cond.raise(input.clone()),
            Some(t) => t
        };

        let from_letter = Col::from_char(from_letter);
        let from_number = Row::from_char(from_number);

        let rest = input.slice_from(post_idx);

        let (to_letter, to_number, _) = match read_square(rest) {
            None => return unreadable_move::cond.raise(input.clone()),
            Some(t) => t
        };

        let to_letter = Col::from_char(to_letter);
        let to_number = Row::from_char(to_number);

        if (from_letter.is_some() && from_number.is_some()
            && to_letter.is_some() && to_number.is_some()) {
            (Square{letter: from_letter.unwrap(),
                    number: from_number.unwrap()},
             Square{letter: to_letter.unwrap(),
                    number: to_number.unwrap()})
        } else {
            unreadable_move::cond.raise(input.clone())
        }
    }

    pub fn antichess() {
        use games::chess::*;
        use std::io;

        let mut b = Game {
            board: chess::initial_board,
            current: white,
            black_taken: ~[], // ~[queen],
            white_taken: ~[], // ~[pawn, pawn]
        };

        let inp = io::stdin();
        loop {
            print(fmt!("%s", b.to_str()));
            let (from, to) = get_move(&b, inp);
            do invalid_move::cond.trap(|(r, m, b)| {
                    let m : Move = m;
                    print(fmt!("invalid move: %s because %s\n",
                               m.to_str(), r.reason()));
                    print("try again");
                    get_move(&b, inp)
                }).inside {
                b.do_move((from, to));
            }
        }
    }
}
