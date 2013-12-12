extern mod extra;
use opts = extra::getopts;
use std::num::One;

fn print_usage(program: &str, _opts: &[opts::Opt]) {
    println!("Usage: {} [options]", program);
    println("--chess");
    println("--antichess");
    println("-h --help\tUsage");
}

fn main() {
    use std::os;

    let opts = ~[opts::optflag("chess"),
                 opts::optopt("antichess"),
                 opts::optflag("h"),
                 opts::optflag("help"),];
    let args = os::args();
    let program = args[0].clone();

    let matches = match opts::getopts(args.tail(), opts) {
        Ok(m) => { m },
        Err(f) => { fail!(f.to_err_msg()) }
    };

    if matches.opt_present("h") || matches.opt_present("help") {
        print_usage(program, opts);
        return;
    }

    if matches.opt_present("chess") {
        games::chess();
    } else {
        games::antichess();
    }
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
        magnitude: u8,
    }

    impl offset {
        fn magnitude(&self) -> u8 { self.magnitude }
    }

    pub fn offset_range(min: char, max: char) -> offset_range {
        offset_range{ min: min, max: max }
    }

    impl offset_range {
        fn inject(&self, c: char) -> Option<offset> {
            if self.min <= c && c <= self.max {
                Some(offset {
                        base: self.min,
                        magnitude: c as u8 - self.min as u8
                    })
            } else {
                None
            }
        }
    }


    pub mod chess {
        use std::util;

        use super::offset_range;
        use N = std::option::None;
        use S = std::option::Some;

        #[deriving(ToStr,Clone,Eq)]
        pub enum Color { black, white }

        impl Color {
            fn rev(self) -> Color {
                match self { black => white, white => black }
            }

            fn pawn_vdir_and_vorigin(self) -> (int, u8) {
                match self {
                    white => (1,1),
                    black => (-1,6)
                }
            }

            fn pawn_end_row(self) -> Row {
                match self {
                    white => Row(7),
                    black => Row(0)
                }
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

        struct OccupiedSquaresIter<'a> {
            board: &'a Board,
            row:Row,
            col:Col,
            color: Color,
        }

        impl<'a> Iterator<Square> for OccupiedSquaresIter<'a> {
            fn next(&mut self) -> Option<Square> {
                loop {
                    let s = Square(self.col, self.row);
                    // debug!("OccupiedSquaresIter step: %s", s.to_str());
                    if *self.row >= 8 {
                        return None;
                    } else if *self.col >= 8 {
                        *self.row = *self.row + 1;
                        *self.col = 0;
                        continue;
                    } else {
                        *self.col = *self.col + 1;
                        match self.board.at(s) {
                            None           => continue,
                            Some(Man(c,_)) => if c == self.color {
                                return Some(s)
                            } else {
                                continue
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

            fn move_is_capturing(&self, (_, to): Move) -> bool {
                self.at(to).is_some()
            }
        }

        #[deriving(Eq)]
        pub struct Row(u8);

        #[deriving(Eq)]
        pub struct Col(u8);

        impl ToStr for Row {
            fn to_str(&self) -> ~str {
                ('1' as u8 + **self).to_str()
            }
        }

        impl Row {
            fn maybe(mag:Option<u8>) -> Option<Row> {mag.map(|m| { Row(m) })}
            pub fn from_char(c:char) -> Option<Row> {
                Row::maybe(offset_range('1', '8').inject(c).map(|x|x.magnitude()))
            }
            fn magnitude(&self) -> u8 { **self }
        }

        impl ToStr for Col {
            fn to_str(&self) -> ~str {
                ('a' as u8 + **self).to_str()
            }
        }

        impl Col {
            fn maybe(mag:Option<u8>) -> Option<Col> {mag.map(|m| { Col(m) })}
            pub fn from_char(c:char) -> Option<Col> {
                Col::maybe(offset_range('a', 'h').inject(c).map(|x|x.magnitude()))
            }
            fn magnitude(&self) -> u8 { **self }
        }

        impl Sub<Col,int> for Col { fn sub(&self, c:&Col) -> int { (**self as int) - (**c as int) } }
        impl Sub<Row,int> for Row { fn sub(&self, c:&Row) -> int { (**self as int) - (**c as int) } }

        #[deriving(Eq)]
        pub struct Square { letter: Col, number: Row }

        fn Square(letter: Col, number: Row) -> Square {
            Square{ letter: letter, number: number }
        }

        pub struct DSquare { dcol: int, drow: int }
        impl Square {
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
                    Some(Square{letter: Col(ncol as u8), number: Row(nrow as u8)})
                }
            }
        }

        impl ToStr for Square {
            fn to_str(&self) -> ~str {
                format!("{:c}{:c}",
                        ('a' as u8 + *self.letter) as char,
                        ('1' as u8 + *self.number) as char)
            }
        }

        impl ToStr for Board {
            fn to_str(&self) -> ~str {
                let mut accum = ~" abcdefgh\n";
                let mut count = 8u;
                for row in self.rows.rev_iter() {
                    accum = accum + format!("{:u}", count);
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
                    accum = accum + format!("{:u}", count) + "\n";
                    count -= 1;
                }
                accum = accum + " abcdefgh";
                format!("{:s}", accum)
            }
        }

        impl Board {
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
                                    continue;
                                } else {
                                    accum.push(s);
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

            fn get_moves_for_pawn(&self, s:Square, color:Color) -> ~[Square] {
                let Square{ letter: _, number: row } = s;

                                let mut accum = ~[];
                                let (vdir, vorigin) = color.pawn_vdir_and_vorigin();

                                let onefwd = s.plus(0, vdir);
                                match onefwd {
                                    None => {},
                                    Some(onefwd) => {
                                        if self.at(onefwd).is_none() {
                                            accum.push(onefwd);
                                            if *row == vorigin {
                                                match onefwd.plus(0, vdir) {
                                                    None => {},
                                                    Some(twofwd) => {
                                                        if self.at(twofwd).is_none() {
                                                            accum.push(twofwd);
                                                        }
                                                    }
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

            pub fn get_moves(&self, s:Square) -> ~[Square] {
                fn flatten(vec: ~[Option<Square>]) -> ~[Square] {
                    vec.flat_map(|x|match *x { None => ~[], Some(o) => ~[o] })
                }
                match self.at(s) {
                    None => ~[],
                    Some(Man(color, piece)) => {
                        let mut blind_moves = match piece {
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
                            pawn   => self.get_moves_for_pawn(s, color)
                        };
                        blind_moves.retain(|to| {
                            match self.at(*to) {
                                None => true,
                                Some(Man(to_color, _)) => color.rev() == to_color
                            }
                        });
                        blind_moves
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

            pub fn do_move_without_validation(&mut self, m: Move, on_promote: Option<Piece>) {
                let (from, to) = m;
                let source_man = self.at(from).unwrap();

                let target_man = match source_man {
                    Man(c, pawn) if to.number == c.pawn_end_row() => Man(c, on_promote.unwrap()),
                    _ => source_man,
                };

                self.clear(from);
                self.put(to, target_man);
            }

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
            move_exposes_king(Move, Square),
            move_is_noncapturing(Move, ~[Move]),
        }

        impl InvalidMoveReason {
            // good enough for now
            pub fn reason(self) -> ~str { self.to_str() }
        }

        // Can respond to an invalid move by providing a different one
        // to apply.
        type invalid_move<'a> = 'a |InvalidMoveReason, Move, &Game| -> Move;

        #[deriving(Clone)]
        pub struct AntichessVariants {
            fewer_pieces_wins_on_stalemate: bool,
            king_has_royal_power: RoyalVariant,
        }

        #[deriving(Clone)]
        pub enum RoyalVariant { NonRoyal, RoyalGettingCheckmatedWins, RoyalGettingCheckmatedLoses, }

        #[deriving(Clone)]
        pub enum RulesVariant { NormalChess, Antichess(AntichessVariants), }

        #[deriving(Clone)]
        pub struct Variant { pawn_promotion: ~[Piece], rules: RulesVariant, }

        #[deriving(Clone)]
        struct Game {
            variant: Variant,
            board: Board,
            current: Color,
            black_taken: ~[Piece],
            white_taken: ~[Piece],
        }

        impl RulesVariant {
            fn king_has_royal_power(self) -> bool {
                match self {
                    NormalChess => true,
                    Antichess(v) => v.king_has_royal_power.king_has_royal_power()
                }
            }
        }

        impl RoyalVariant {
            fn king_has_royal_power(self) -> bool {
                match self {
                    NonRoyal => false,
                    RoyalGettingCheckmatedLoses => true,
                    RoyalGettingCheckmatedWins => true,
                }
            }
        }

        pub fn pieces_to_str(c: Color, v: &[Piece]) -> ~str {
            v.map(|p| Man(c,*p).to_str()).concat()
        }

        struct ElaboratedMove {
            move: Move,
            source: Man,
            target: Option<Man>,
        }

        // AllMovesIter includes moves that capture the king and
        // *also* moves that leave the current player's king exposed
        // to capture.  You need to do a post-psss over the reuslt to
        // ensure that you have not put your own king into check.
        struct AllMovesIter<'a> {
            board: &'a Board,
            current: Color,
            rest_squares: OccupiedSquaresIter<'a>,
            curr_moves: Option<(Square, ~[Square], uint)>,
        }

        enum CapturingFilter { IgnoreCapturing, CapturingOnly, NoncapturingOnly, }

        struct MovesFilterLegality<'a> {
            iter: AllMovesIter<'a>,
            capturing_filter: CapturingFilter,
        }

        impl<'a> Iterator<Move> for AllMovesIter<'a> {
            fn next(&mut self) -> Option<Move> {
                // debug!("AllMovesIter next: %?", self);

                enum Step {
                    Yielding(uint, Move),
                    LoopNextSquare,
                    LoopNewIndex(uint),
                    LoopNewVec(Square, ~[Square]),
                    Finished
                }

                impl ToStr for Step {
                    fn to_str(&self) -> ~str {
                        match *self {
                            Yielding(n, m)    => format!("Yielding({:u}, {:s})", n, m.to_str()),
                            LoopNextSquare    => format!("LoopNextSquare"),
                            LoopNewIndex(n)   => format!("LoopNewIndex({:u})", n),
                            LoopNewVec(ref s, ref ss) =>
                                format!("LoopNewVec({:s}, {:s})", s.to_str(), ss.to_str()),
                            Finished          => format!("Finished")
                        }
                    }
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
                    // debug!("AllMovesIter step: {:s}", step.to_str());
                    match step {
                        Yielding(j, m)     => {
                            match curr_moves {
                                Some((f, v, _)) => self.curr_moves = Some((f, v, j)),
                                None => self.curr_moves = curr_moves,
                            }
                            return Some(m)
                        },
                        LoopNextSquare  => { curr_moves = None; continue; },
                        LoopNewIndex(i) => {
                            let (from, to_vec, _) = curr_moves.unwrap();
                            curr_moves = Some((from, to_vec, i));
                            continue;
                        },
                        LoopNewVec(s, v)  => {
                            curr_moves = Some((s, v, 0));
                            continue;
                        },
                        Finished       => { self.curr_moves = curr_moves; return None },
                    }
                }
            }
        }

        impl<'a> AllMovesIter<'a> {
            fn filter_exposed_kings(self, f: CapturingFilter) -> MovesFilterLegality<'a> {
                MovesFilterLegality {
                    iter: self, capturing_filter: f,
                }
            }
        }

        fn all_moves_iter_for<'r>(board: &'r Board, player: Color) -> AllMovesIter<'r> {
            let squares = board.occupied_squares_iter(player);
            AllMovesIter {
                board: board, current: player, rest_squares: squares, curr_moves: None,
            }
        }

        fn king_capturable<'r>(board: &'r Board, target: Color) -> Option<Square> {
            for (from, to) in all_moves_iter_for(board, target.rev()) {
                match board.at(to) {
                    Some(Man(_, king)) => return Some(from),
                    _ => {}
                }
            }
            return None;
        }

        impl<'a> Iterator<Move> for MovesFilterLegality<'a> {
            fn next(&mut self) -> Option<Move> {
                for m in self.iter {
                    match self.capturing_filter {
                        CapturingOnly if ! self.iter.board.move_is_capturing(m) => continue,
                        NoncapturingOnly if self.iter.board.move_is_capturing(m) => continue,
                        _ => {},
                    }

                    let mut b = self.iter.board.clone();
                    b.do_move_without_validation(m, Some(queen));
                    if king_capturable(&b, self.iter.current).is_some() {
                        continue
                    } else {
                        return Some(m);
                    }
                }
                None
            }
        }

        enum GameEnd {
            CheckMateFor(Color),
            StaleMate,
            AllPiecesGone,
            AllPiecesButKingGone,
        }

        

        impl Game {
            pub fn to_str(&self) -> ~str {
                let ret = self.board.to_str();
                let ret = ret.replace("8\n",
                                      format!("8 {:s} \n", pieces_to_str(black, self.black_taken)));
                let ret = ret.replace("1\n",
                                      format!("1 {:s} \n", pieces_to_str(white, self.white_taken)));
                let ret = ret + "\n" + self.current.to_str() + "'s move";
                ret
            }

            pub fn is_check(&self) -> bool {
                king_capturable(&self.board, self.current).is_some()
            }

            pub fn king_has_royal_power(&self) -> bool {
                self.variant.rules.king_has_royal_power()
            }

            pub fn is_capturing_compulsory(&self) -> bool {
                match self.variant.rules {
                    NormalChess => false,
                    Antichess(_) => true,
                }
            }

            pub fn all_pieces_gone(&self) -> bool {
                self.board.occupied_squares_iter(self.current).next().is_none()
            }

            pub fn all_pieces_but_king_gone(&self) -> bool {
                let mut i = self.board.occupied_squares_iter(self.current);
                match i.next() {
                    None => false,
                    Some(s) => i.next().is_none() && match self.board.at(s).unwrap() {
                        Man(_, king) => true,
                        Man(_, _) => false,
                    }
                }
            }

            pub fn is_game_over(&self) -> Option<GameEnd> {
                let current = self.current;
                fn king_capturable_in_all_moves(g:&Game, current:Color) -> bool {
                    g.all_moves_iter().all(|move| {
                        let mut b = g.board.clone();
                        b.do_move_without_validation(move, Some(queen));
                        king_capturable(&b, current).is_some()
                    })
                }

                if self.king_has_royal_power() &&
                    king_capturable_in_all_moves(self, current) {
                    if king_capturable(&self.board, current).is_some() {
                        Some(CheckMateFor(self.current.rev()))
                    } else {
                        Some(StaleMate)
                    }
                } else {
                    match self.variant.rules {
                        NormalChess => None,
                        Antichess(v) => match v.king_has_royal_power {
                            NonRoyal => if self.all_pieces_gone() {
                                Some(AllPiecesGone)
                            } else {
                                None
                            },
                            RoyalGettingCheckmatedWins | RoyalGettingCheckmatedLoses =>
                                if self.all_pieces_but_king_gone() {
                                Some(AllPiecesButKingGone)
                            } else {
                                None
                            }
                        }
                    }
                }
            }

            fn all_moves_iter_core<'r>(&'r self, f:CapturingFilter) -> MovesFilterLegality<'r> {
                all_moves_iter_for(&'r self.board, self.current).filter_exposed_kings(f)
            }

            pub fn capturing_moves_iter<'r>(&'r self) -> MovesFilterLegality<'r> {
                self.all_moves_iter_core(CapturingOnly)
            }

            pub fn noncapturing_moves_iter<'r>(&'r self) -> MovesFilterLegality<'r> {
                self.all_moves_iter_core(NoncapturingOnly)
            }

            pub fn all_moves_iter<'r>(&'r self) -> MovesFilterLegality<'r> {
                self.all_moves_iter_core(IgnoreCapturing)
            }

            pub fn moves_iter<'r>(&'r self) -> MovesFilterLegality<'r> {
                match self.variant.rules {
                    NormalChess => self.all_moves_iter(),
                    Antichess(_) => {
                        if self.capturing_moves_iter().next().is_none() {
                            self.noncapturing_moves_iter()
                        } else {
                            self.capturing_moves_iter()
                        }
                    }
                }
            }

            pub fn is_illegal(&self, m: Man, from: Square, to: Square) -> Option<~str> {
                // Redundantly checking that move is non-trival.
                if from == to { return Some(~"cannot make trivial move"); }

                let Man(color, p) = m;
                let dsquare = to.delta_from(from);
                let DSquare{ dcol: dcol, drow: drow } = dsquare;
                debug!("is_illegal m: {:s} (from, to): {:s} dsquare: {:?}",
                       m.to_str(), (from,to).to_str(), dsquare);
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
                        let (vdir, vorigin) = color.pawn_vdir_and_vorigin();
                        debug!("is_illegal pawn vdir: {:?} vorigin: {:?} from.number: {:?} dcol: {:?} drow: {:?}",
                               vdir, vorigin, *from.number, dcol, drow);
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
                            Some(format!("Pawn in home row move forward by one or two spaces, not {:d}",
                                      drow))
                        } else if *from.number != vorigin && drow != vdir {
                            Some(format!("Pawn outside home row move forward by one space, not {:d}",
                                      drow))
                        } else if self.board.at(to).is_some() && dcol == 0 {
                            Some(~"Pawns cannot capture unless moving diagonally")
                        } else {
                            Some(~"Whatever that move is, a pawn cannot do it")
                        }
                    }
                }
            }

            pub fn find_occupied_between(&self, from: Square, to: Square) -> Option<Square> {
                fn signum(from:u8, to:u8) -> int {
                    if to > from { 1 } else if to == from { 0 } else { -1 }
                }

                let (col, end_col) = (*from.letter, *to.letter);
                let (row, end_row) = (*from.number, *to.number);

                let dcol = signum(col, end_col);
                let drow = signum(row, end_row);

                let mut row = (row as int + drow) as u8;
                let mut col = (col as int + dcol) as u8;
                loop {
                    if row == end_row && col == end_col { return None; }
                    let s = Square{ letter: Col(col), number: Row(row) };
                    if self.board.at(s).is_some() {
                        return Some(s);
                    }
                    row = (row as int + drow) as u8;
                    col = (col as int + dcol) as u8;
                }
            }

            pub fn check_for_blockage(&self, m: Man, from: Square, to: Square)
                -> Option<Square> {
                let Man(_, p) = m;
                match p {
                    king  | knight | pawn   => None,
                    queen | rook   | bishop => self.find_occupied_between(from, to),
                }
            }

            

            pub fn validate_move<'a>(&self, move: Move, handler: &invalid_move<'a>)
                                     -> ElaboratedMove {
                let mut move = move;

                let raise = |reason| {
                    (*handler)(reason, move, self)
                };

                // macro_rules! retry(
                //     ($reason:expr) => { move = raise($reason); loop; }
                // );
                // 
                // #[cfg(try_macros)]
                // macro_rules! retry2(
                //     ($d:ident, $reason:expr) => { $d = raise($reason); loop; }
                // );

                loop {
                    let (from, to) = move;

                    if from == to {
                        let r = source_and_target_are_same_square(from);
                        // Neither of the below macros work.  It is not clear why.
                        // (I can expose via the input sequence: "c2 c2", "c2 c3")
                        // retry!(move, r)
                        // retry2!(move, r)
                        move = raise(r); continue;
                    }

                    let m = *self.board.cell(from);

                    debug!("validate move {:?} : man {:?}", move.to_str(), m);

                    match m {
                        None => {
                            move = raise(source_square_is_empty(from)); continue;
                        }
                        Some(source_man @ Man(color, _)) => {
                            if self.current != color {
                                move = raise(piece_is_not_your_color(source_man)); continue;
                            }

                            match self.is_illegal(source_man, from, to) {
                                None => {},
                                Some(text) => {
                                    let r = man_cannot_make_move(source_man, move, text);
                                    move = raise(r); continue;
                                }
                            }

                            match self.check_for_blockage(source_man, from, to) {
                                None => {},
                                Some(sq) => { move = raise(move_is_blocked_by(move, sq)); continue; },
                            }

                            debug!("validate move {:?} : source {:?} to {:?}",
                                   move.to_str(), source_man, to.to_str());
                            let target_man : Option<Man> =
                                match *self.board.cell(to) {
                                // empty target is a-okay
                                None => {
                                    None
                                },
                                Some(man @ Man(color, _)) => {
                                    debug!("validate move {:?} : target man {:?}", move.to_str(), man);
                                    if self.current == color {
                                        let r = target_square_holds_piece_of_your_color(man);
                                        move = raise(r); continue;
                                    } else {
                                        Some(man)
                                    }
                                },
                            };

                            if self.king_has_royal_power() {
                                let mut b = self.board.clone();
                                b.do_move_without_validation(move, Some(queen));
                                match king_capturable(&b, self.current) {
                                    Some(s) => { move = raise(move_exposes_king(move, s)); continue },
                                    _ => {}
                                }
                            }

                            if self.is_capturing_compulsory() && target_man.is_none() {
                                // confirm that no capturing move exists.
                                let capturing_moves : ~[Move] =
                                    self.capturing_moves_iter().collect();
                                if capturing_moves.len() > 0 {
                                    let exn = move_is_noncapturing(move, capturing_moves);
                                    move = raise(exn); continue;
                                }
                            }

                            // Okay. now everything has been checked.
                            return ElaboratedMove {
                                move: move,
                                source: source_man,
                                target: target_man
                            };
                        }
                    }
                }
            }

            pub fn do_move<'a>(&mut self, move: Move, handler: &invalid_move<'a>) {
                let ElaboratedMove{
                    move: move, source: _, target: target_man
                } = self.validate_move(move, handler);

                self.board.do_move_without_validation(move, Some(queen));

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
                //a      b      c      d      e      f      g      h
                [[S(wr), S(wn), S(wb), S(wq), S(wk), S(wb), S(wn), S(wr)], //1
                 [S(wp), S(wp), S(wp), S(wp), S(wp), S(wp), S(wp), S(wp)], //2
                 [N,     N,     N,     N,     N,     N,     N,     N    ], //3
                 [N,     N,     N,     N,     N,     N,     N,     N    ], //4
                 [N,     N,     N,     N,     N,     N,     N,     N    ], //5
                 [N,     N,     N,     N,     N,     N,     N,     N    ], //6
                 [S(bp), S(bp), S(bp), S(bp), S(bp), S(bp), S(bp), S(bp)], //7
                 [S(br), S(bn), S(bb), S(bq), S(bk), S(bb), S(bn), S(br)]],//8
        };
    }

    type ChessMove = chess::Move;
    type retry<'a, BUF> = 'a |input: ~str, p: &mut MoveReader<'a, BUF>| -> ChessMove;

    type no_input<'a, BUF> = 'a |&mut MoveReader<'a, BUF>| -> ChessMove;

    struct Retry<'a, BUF> {
        attempts_since_success: u32,
        call: retry<'a, BUF>,
        no_input: no_input<'a, BUF>,
    }

    impl<'a, BUF:Buffer> Retry<'a, BUF> {
        fn new(call: retry<'a, BUF>, no_input: no_input<'a, BUF>) -> Retry<'a, BUF> {
            Retry { attempts_since_success: 0, call: call, no_input: no_input }
        }
    }

    struct MoveReader<'a, BUF> {
        reparse: Retry<'a, BUF>,
    }

    impl<'a, BUF:Buffer> MoveReader<'a, BUF> {
        fn reparse(&mut self, input: ~str) -> ChessMove {
            self.reparse.attempts_since_success += 1;
            (self.reparse.call)(input, self)
        }
        fn no_input(&mut self) -> ChessMove {
            self.reparse.attempts_since_success += 1;
            (self.reparse.no_input)(self)
        }
    }
    pub fn get_move_recur<BUF:Buffer>(g: &chess::Game, inp: &mut BUF) -> ChessMove {
        let retry = |input: ~str, p: &mut MoveReader<BUF>| {
            println(format!("Attempt {}: Could not parse input: <<{:s}>>",
                            p.reparse.attempts_since_success,
                            input.trim()));
            println("try again");
            print(format!("{:s}", g.to_str()));

            // XXX Rust is not tail-recursive, nor is this a
            // tail-recursive context.  So this pattern is bad.
            // However, (1.) we don't expect the user to make an
            // unbounded number of input errors, and (2.) if we
            // did expect that, we could keep a count and fail
            // after some attempt threshold.
            //
            // (The fix is probably to rewrite parse_move to
            // return Option instead of invoking a retry-handler;
            // its not like parse code ever attempts to do
            // anything non-trivial with value produced by retry.)
            p.get_move_recur(g, inp)
        };

        let fail_on_no_input = |p: &mut MoveReader<BUF>| {
            fail!("Attempt {}: no input provided, failing",
                  p.reparse.attempts_since_success);
        };

        let mut p = MoveReader { reparse: Retry::new(retry, fail_on_no_input) };
        p.get_move_recur(g, inp)
    }

    impl<'a, BUF:Buffer> MoveReader<'a, BUF> {
        fn get_move_recur(&mut self, g: &chess::Game, inp:&mut BUF) -> ChessMove {
            print(format!("{:s}'s move? ", g.current.to_str()));
            let input = inp.read_line();
            println("");
            debug!("Got input: {:?}",  input);

            self.parse_opt_move(input.clone())
        }
    }

    pub fn get_move<BUF:Buffer>(g: &chess::Game, inp: &mut BUF) -> ChessMove {
        match g.current {
            chess::black => {
                let mut i = g.moves_iter();
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

    impl<'a, BUF:Buffer> MoveReader<'a, BUF> {
    pub fn read_square(&self, input: &str) -> Option<(char, char, uint)> {
        let idx = input.find(|c| ('a' <= c && c <= 'h'));
        debug!("read_square idx: {:?}", idx);
        match idx {
            None => None,
            Some(idx) => {
                if idx+1 >= input.char_len() {
                    None
                } else {
                    let to_letter = input.char_at(idx);
                    let to_number = input.char_at(idx+1);
                    let val = (to_letter, to_number, idx+2);
                    debug!("read_square val: {:?}", val);
                    Some(val)
                }
            }
        }
    }

    pub fn parse_opt_move(&mut self, input: Option<~str>) -> ChessMove {
        let ret = match input {
            Some(s) => self.parse_move(s),
            None    => self.no_input(),
        };
        self.reparse.attempts_since_success = 0;
        ret
    }

    pub fn parse_move(&mut self, input: ~str) -> ChessMove {
        use ch = games::chess;

        if input.char_len() < 2 {
            return self.reparse(input.clone());
        }

        let (from_letter, from_number, post_idx) = match self.read_square(input) {
            None => return self.reparse(input.clone()),
            Some(t) => t
        };

        let from_letter = ch::Col::from_char(from_letter);
        let from_number = ch::Row::from_char(from_number);

        let rest = input.slice_from(post_idx);

        let (to_letter, to_number, _) = match self.read_square(rest) {
            None => return self.reparse(input.clone()),
            Some(t) => t
        };

        let to_letter = ch::Col::from_char(to_letter);
        let to_number = ch::Row::from_char(to_number);

        if (from_letter.is_some() && from_number.is_some()
            && to_letter.is_some() && to_number.is_some()) {
            (ch::Square{letter: from_letter.unwrap(), number: from_number.unwrap()},
             ch::Square{letter:   to_letter.unwrap(), number: to_number.unwrap()})
        } else {
            self.reparse(input.clone())
        }
    }
    }

    pub fn chess() {
        use ch = games::chess;
        let v = ch::Variant{
            pawn_promotion: ~[ch::queen, ch::rook, ch::bishop, ch::knight],
            rules: ch::NormalChess,
        };

        chess_game(v);
    }

    pub fn antichess() {
        use ch = games::chess;
        let v = ch::Variant{
            pawn_promotion: ~[ch::queen, ch::rook, ch::bishop, ch::knight],
            rules: ch::Antichess(ch::AntichessVariants{
                    fewer_pieces_wins_on_stalemate: false,
                    king_has_royal_power: ch::RoyalGettingCheckmatedLoses,
                })};

        chess_game(v);
    }

    pub fn chess_game(variant: chess::Variant) {
        use ch = games::chess;
        use std::io;
        use std::io::buffered::BufferedReader;

        let mut b = ch::Game {
            variant: variant,
            board: ch::initial_board,
            current: ch::white,
            black_taken: ~[], // ~[queen],
            white_taken: ~[], // ~[pawn, pawn]
        };

        let inp = io::stdin();
        let mut inp = BufferedReader::new(inp);

        loop {
            println(format!("{:s}", b.to_str()));

            match b.is_game_over() {
                None => {},
                Some(end) => { println(format!("{:?}", end)); break; }
            }

            if b.is_check() { println("Check"); }

            print("moves: ");
            for m in b.moves_iter() {
                let (from, to) = m;
                print(format!("({:s}, {:s})", from.to_str(), to.to_str()));
            }
            println("");


            let (from, to) = get_move(&b, &mut inp);

            let handler = |r:ch::InvalidMoveReason, m:ch::Move, b:&ch::Game| {
                print(format!("invalid move: {:s} because {:s}\n",
                                  m.to_str(), r.reason()));
                print("try again, ");
                get_move(b, &mut inp)
            };
            b.do_move((from, to), &handler);
        }
    }
}
