use std::num::One;

fn main() {
    games::antichess();
}

 /// A range of numbers from [0, N)
#[deriving(Clone, DeepClone)]
pub struct MyRange<A> {
    priv state: A,
    priv stop: A,
    priv one: A
}

impl<A: Add<A, A> + Ord + Clone> Iterator<A> for MyRange<A> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.state < self.stop {
            let result = self.state.clone();
            self.state = self.state + self.one;
            Some(result)
        } else {
            None
        }
    }

    // FIXME: #8606 Implement size_hint() on Range
    // Blocked on #8605 Need numeric trait for converting to `Option<uint>`
}

// GRRR. iterator library does not offer a way to reuse the existing Range and plugging in
// one's one One (because all of the fields are private); you can only inherit preexisting
// settings for One.
pub fn range_inclusive<A: Add<A, A> + Ord + Clone + One + Neg<A>>(start: A, stop: A) -> MyRange<A> {
    if start <= stop {
        MyRange{state: start, stop: stop, one: One::one()}
    } else {
        let pos_one : A = One::one();
        let neg_one = pos_one.neg();
        MyRange{state: start, stop: stop, one: neg_one }
    }
}

pub mod games {
    use std::rand::random;

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
        use std::util;
        use super::offset_range;

        #[deriving(ToStr,Clone,Eq)]
        pub enum Color { black, white }

        impl Color {
            fn rev(self) -> Color {
                match self { black => white, white => black }
            }
        }

        #[deriving(Clone, ToStr)]
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

        struct OccupiedSquaresIter<'self> {
            board: &'self Board,
            row:Row,
            col:Col,
            color: Color,
        }

        impl<'self> Iterator<Square> for OccupiedSquaresIter<'self> {
            fn next(&mut self) -> Option<Square> {
                debug!("OccupiedSquaresIter next: %?", (self.row, self.col));
                loop {
                    debug!("OccupiedSquaresIter step: %?", (self.row, self.col));
                    if *self.row >= 8 {
                        return None;
                    } else if *self.col >= 8 {
                        *self.row = *self.row + 1;
                        *self.col = 0;
                        loop;
                    } else {
                        let s = Square { letter: self.col, number: self.row };
                        *self.col = *self.col + 1;
                        match self.board.at(s) {
                            None           => loop,
                            Some(Man(c,_)) => if c == self.color {
                                return Some(s)
                            } else {
                                loop
                            },
                        }
                    }
                }
            }
        }

        impl Board {
            fn occupied_squares_iter<'a>(&'a self, color: Color) -> OccupiedSquaresIter<'a> {
                OccupiedSquaresIter { board: self, row: Row(0), col: Col(0), color: color }
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

        impl Sub<Col,int> for Col { fn sub(&self, c:&Col) -> int { (**self as int) - (**c as int) } }
        impl Sub<Row,int> for Row { fn sub(&self, c:&Row) -> int { (**self as int) - (**c as int) } }

        #[deriving(Eq)]
        pub struct Square { letter: Col, number: Row }

        pub struct DSquare { dcol: int, drow: int }
        impl Square {
            fn delta_to(&self, s: Square) -> DSquare {
                DSquare{ dcol: s.letter - self.letter, drow: s.number - self.number }
            }
            fn delta_from(&self, s: Square) -> DSquare {
                DSquare{ dcol: self.letter - s.letter, drow: self.number - s.number }
            }
            fn plus(&self, dcol: int, drow: int) -> Option<Square> {
                let Square{ letter: col, number: row } = *self;
                let ncol = *col as int + dcol;
                let nrow = *row as int + drow;
                if ncol < 0 || ncol >= 8 || nrow < 0 || nrow >= 8 {
                    None
                } else {
                    Some(Square{letter: Col(ncol as uint), number: Row(nrow as uint)})
                }
            }
        }

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

            pub fn radiate(&self, s:Square, units:~[(int,int)]) -> ~[Square] {
                let mut accum = ~[];
                for v in units.iter() {
                    let (dc, dr) = *v;
                    let mut cursor = s;
                    loop {
                        let c = cursor.plus(dc,dr);
                        match c {
                            Some(s) => {
                                if self.at(s).is_none() {
                                    accum.push(s);
                                    cursor = s;
                                    loop;
                                } else {
                                    break;
                                }
                            },
                            _ => break,
                        }
                    }
                }
                return accum;
            }
            pub fn radiate_diagonal(&self, s:Square) -> ~[Square] {
                self.radiate(s, ~[(-1,-1), (-1,1), (1,1), (1,-1)])
            }

            pub fn radiate_linear(&self, s:Square) -> ~[Square] {
                self.radiate(s, ~[(-1,0), (0,-1), (0,1), (1,0)])
            }

            pub fn radiate_eightway(&self, s:Square) -> ~[Square] {
                self.radiate(s, ~[(-1,-1), (-1,1), (1,1), (1,-1),
                                  (-1,0), (0,-1), (0,1), (1,0)])
            }

            pub fn get_moves(&self, s:Square) -> ~[Square] {
                fn flatten(vec: ~[Option<Square>]) -> ~[Square] {
                    vec.flat_map(|x|match *x { None => ~[], Some(o) => ~[o] })
                }
                let Square{ letter: _, number: row } = s;
                match self.at(s) {
                    None => ~[],
                    Some(Man(color, piece)) => {
                        match piece {
                            king   => flatten(~[s.plus(-1,-1), s.plus(-1,0), s.plus(-1,1),
                                                s.plus( 0,-1),               s.plus( 0,1),
                                                s.plus( 1,-1), s.plus( 1,0), s.plus( 1,1)]),
                            queen  => self.radiate_eightway(s),
                            rook   => self.radiate_linear(s),
                            bishop => self.radiate_diagonal(s),
                            knight => flatten(~[s.plus(-2,-1), s.plus(-2, 1),
                                                s.plus(-1,-2), s.plus(-1, 2),
                                                s.plus( 1,-2), s.plus( 1, 2),
                                                s.plus( 2,-1), s.plus( 2, 1)]),
                            pawn   => {
                                let mut accum = ~[];
                                let (vdir, vorigin) = match color {
                                    white => (1,1),
                                    black => (-1,6)
                                };

                                let onefwd = s.plus(0, vdir);
                                match onefwd {
                                    None => {},
                                    Some(onefwd) => {
                                        if self.at(onefwd).is_none() {
                                            accum.push(onefwd);
                                            if *row == vorigin {
                                                match onefwd.plus(0, vdir) {
                                                    None => {},
                                                    Some(twofwd) => accum.push(twofwd),
                                                }
                                            }
                                        }
                                    }
                                }
                                match s.plus(-1, vdir) {
                                    None => {},
                                    Some(lft) => {
                                        if self.at(lft).is_some() {
                                            accum.push(lft);
                                        }
                                    }
                                }
                                match s.plus( 1, vdir) {
                                    None => {},
                                    Some(rgt) => {
                                        if self.at(rgt).is_some() {
                                            accum.push(rgt);
                                        }
                                    }
                                }
                                accum
                            }
                        }
                    }
                }
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
            man_cannot_make_move(Man, Move, ~str),
            move_is_blocked_by(Move, Square),
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

        struct ElaboratedMove {
            move: Move,
            source: Man,
            target: Option<Man>,
        }

        struct AllMovesIter<'self> {
            board: &'self Board,
            current: Color,
            rest_squares: OccupiedSquaresIter<'self>,
            curr_moves: Option<(Square, ~[Square], uint)>,
        }

        impl<'self> Iterator<Move> for AllMovesIter<'self> {
            fn next(&mut self) -> Option<Move> {
                debug!("AllMovesIter next: %?", self);
                enum Step {
                    Yielding(uint, Move),
                    LoopNextSquare,
                    LoopNewIndex(uint),
                    LoopNewVec(Square, ~[Square]),
                    Finished
                }
                let mut curr_moves = None;
                util::swap(&mut self.curr_moves, &mut curr_moves);

                loop {
                    let step = match curr_moves {
                        Some((from, ref to_vec, idx)) => {
                            if idx < to_vec.len() {
                                let s = to_vec[idx];
                                match self.board.at(s) {
                                    None              => { Yielding(idx+1, (from,s)) },
                                    Some(Man(col, _)) => if col != self.current {
                                        Yielding(idx+1, (from,s))
                                    } else {
                                        LoopNewIndex(idx+1)
                                    }
                                }
                            } else {
                                LoopNextSquare
                            }
                        },
                        None => match self.rest_squares.next() {
                            Some(s) => { LoopNewVec(s, self.board.get_moves(s)) }
                            None => Finished
                        }
                    };
                    debug!("AllMovesIter step: %?", step);
                    match step {
                        Yielding(j, m)     => {
                            match curr_moves {
                                Some((f, v, _)) => self.curr_moves = Some((f, v, j)),
                                None => self.curr_moves = curr_moves,
                            }
                            return Some(m)
                        },
                        LoopNextSquare  => { curr_moves = None; loop; },
                        LoopNewIndex(i) => {
                            let (from, to_vec, _) = curr_moves.unwrap();
                            curr_moves = Some((from, to_vec, i));
                            loop;
                        },
                        LoopNewVec(s, v)  => {
                            curr_moves = Some((s, v, 0));
                            loop;
                        },
                        Finished       => { self.curr_moves = curr_moves; return None },
                    }
                    util::unreachable() // is there a static_fail?
                }
            }
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

            pub fn all_moves_iter<'r>(&'r self) -> AllMovesIter<'r> {
                AllMovesIter {
                    board: &'self self.board,
                    current: self.current,
                    rest_squares: self.board.occupied_squares_iter(self.current),
                    curr_moves: None,
                }
            }

            // XXX bleah, I would prefer to pass &self here, not &mut self, but this is
            // intended to be called from validate_move.  (Which, come to think of it,
            // might also take &self instead of &mut self?)  Still, it seems like I keep
            // encountering motivations for &const self or some such.
            pub fn is_illegal(&mut self, m: Man, from: Square, to: Square) -> Option<~str> {
                if from == to { return None; } // Redundantly checking that move is non-trival.
                let Man(color, p) = m;
                let dsquare = to.delta_from(from);
                let DSquare{ dcol: dcol, drow: drow } = dsquare;
                debug!("is_illegal m: %? (from, to): %? dsquare: %?", m, (from,to), dsquare);
                match p {
                    king   => if dcol.abs() <= 1 && drow.abs() <= 1 { None }
                        else { Some(~"kings can only move one spot") },
                    queen  => if drow == 0 || dcol == 0 || drow.abs() == dcol.abs() { None }
                        else { Some(~"queens must move in straight line") },
                    rook   => if drow == 0 || dcol == 0 { None }
                        else { Some(~"rooks must move by row or by column alone") },
                    bishop => if drow.abs() == dcol.abs() { None }
                        else { Some(~"bishops must move diagonally") },
                    knight =>
                        if ((drow.abs() == 2 && dcol.abs() == 1) ||
                            (drow.abs() == 1 && dcol.abs() == 2)) { None }
                        else { Some(~"knights move by two and then by one") },
                    pawn   => {
                        let (vdir, vorigin) = match color { white => (1,1), black => (-1,6) };

                        if *from.number == vorigin && dcol == 0 && drow == vdir*2
                            && self.board.at(to).is_none() {
                            None // foward by two from home row
                        } else if dcol == 0 && drow == vdir && self.board.at(to).is_none() {
                            None // foward by one
                        } else if dcol.abs() == 1 && drow == vdir && self.board.at(to).is_some() {
                            None // diagonal capture
                        } else if drow*vdir < 0 {
                            Some(~"Pawns cannot move backwards")
                        } else if dcol.abs() > 1 || (dcol.abs() == 1 && drow != vdir) {
                            Some(~"Pawns cannot capture except via immediate diagonal")
                        } else if dcol.abs() == 1 && drow == vdir && self.board.at(to).is_none() {
                            Some(~"Pawns cannot move diagonally unless capturing.")
                        } else if (*from.number == vorigin && dcol == 0 && drow.abs() > 2) {
                            Some(fmt!("Pawn in home row move forward by one or two spaces, not %d",
                                      drow))
                        } else if drow != vdir {
                            Some(fmt!("Pawn outside home row move forward by one space, not %d",
                                      drow))
                        } else if self.board.at(to).is_some() && dcol == 0 {
                            Some(~"Pawns cannot capture unless moving diagonally")
                        } else {
                            Some(~"Whatever that move is, a pawn cannot do it")
                        }
                    }
                }
            }

            pub fn find_occupied_between(&mut self, from: Square, to: Square) -> Option<Square> {
                fn signum(from:uint, to:uint) -> int {
                    if to > from { 1 } else if to == from { 0 } else { -1 }
                }

                let (col, end_col) = (*from.letter, *to.letter);
                let (row, end_row) = (*from.number, *to.number);

                let dcol = signum(col, end_col);
                let drow = signum(row, end_row);

                let mut row = (row as int + drow) as uint;
                let mut col = (col as int + dcol) as uint;
                loop {
                    if row == end_row && col == end_col { return None; }
                    let s = Square{ letter: Col(col), number: Row(row) };
                    if self.board.at(s).is_some() {
                        return Some(s);
                    }
                    row = (row as int + drow) as uint;
                    col = (col as int + dcol) as uint;
                }
            }

            pub fn check_for_blockage(&mut self, m: Man, from: Square, to: Square)
                -> Option<Square> {
                let Man(_, p) = m;
                match p {
                    king  | knight | pawn   => None,
                    queen | rook   | bishop => self.find_occupied_between(from, to),
                }
            }

            pub fn validate_move(&mut self, move: Move) -> ElaboratedMove {
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

                            match self.is_illegal(source_man, from, to) {
                                None => {},
                                Some(text) => {
                                    let r = man_cannot_make_move(source_man, move, text);
                                    move = raise(r); loop;
                                }
                            }

                            match self.check_for_blockage(source_man, from, to) {
                                None => {},
                                Some(sq) => { move = raise(move_is_blocked_by(move, sq)); loop; },
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
                                        let r = target_square_holds_piece_of_your_color(man);
                                        move = raise(r); loop;
                                    } else {
                                        Some(man)
                                    }
                                },
                            };

                            // Okay. now everything has been checked.
                            return ElaboratedMove {
                                move: move,
                                source: source_man,
                                target: target_man
                            };
                        }
                    }
                    fail!("should never get here"); // is there a static_fail?
                }
            }

            pub fn do_move(&mut self, (from, to): Move) {
                let mut move = (from, to);

                let ElaboratedMove{
                    move: move, source: source_man, target: target_man
                } = self.validate_move(move);

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
                [[Some(wr), Some(wn), Some(wb), Some(wq), Some(wk), Some(wb), Some(wn), Some(wr)],
                 [Some(wp), Some(wp), Some(wp), Some(wp), Some(wp), Some(wp), Some(wp), Some(wp)],
                 [None,     None,     None,      None,     None,    None,     None,     None    ],
                 [None,     None,     None,      None,     None,    None,     None,     None    ],
                 [None,     None,     None,      None,     None,    None,     None,     None    ],
                 [None,     None,     None,      None,     None,    None,     None,     None    ],
                 [Some(bp), Some(bp), Some(bp), Some(bp), Some(bp), Some(bp), Some(bp), Some(bp)],
                 [Some(br), Some(bn), Some(bb), Some(bq), Some(bk), Some(bb), Some(bn), Some(br)]],
        };
    }

    condition! {
        pub unreadable_move : ~str -> (super::chess::Square, super::chess::Square);
    }

    type ChessMove = chess::Move;

    pub fn get_move_recur(g: &chess::Game, inp: @Reader) -> ChessMove {
        print(fmt!("%s's move? ", g.current.to_str()));
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
        match g.current {
            chess::black => {
                let mut i = g.all_moves_iter();
                let mut m : ChessMove = i.next().unwrap();
                loop {
                    match i.next() {
                        None => return m,
                        Some(new) => {
                            let b : bool = random();
                            if b { return m; }
                            m = new;
                        }
                    }
                }
            },
            chess::white => { get_move_recur(g, inp) },
        }
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
            (Square{letter: from_letter.unwrap(), number: from_number.unwrap()},
             Square{letter:   to_letter.unwrap(), number: to_number.unwrap()})
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
            println(fmt!("%s", b.to_str()));
            for m in b.all_moves_iter() {
                let (from, to) = m;
                print(fmt!("(%s, %s)", from.to_str(), to.to_str()));
            }
            println("");
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
