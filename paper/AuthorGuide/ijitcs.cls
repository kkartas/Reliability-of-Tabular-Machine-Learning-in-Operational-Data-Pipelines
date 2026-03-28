\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ijitcs}[2023/11/18]

%% Article options
\DeclareOption{10pt}{
	\PassOptionsToClass{\CurrentOption}{article}
}

%% Process given options
\ProcessOptions\relax

\LoadClass[twoside,10pt]{article}


\usepackage[a4paper]{geometry}



\geometry{%
	top = 3cm,
	bottom = 2cm,
	left = 2cm,
	right = 2cm,
	% showframe,
	headsep = .5cm,
	headheight = 50pt,
	footskip = 18pt,
}
\def\loadmaingeometry{%
	\newgeometry{%
		top = 3cm,
		bottom = 2cm,
		left = 2cm,
		right = 2cm,
		% showframe,
		headsep = 1.33cm,
		footskip = .65cm
	}
}




\usepackage{times}





\ExplSyntaxOn
% \clist_map_inline:nn{volume,month,year,doi,startpage,endpage}{

% }

\keys_define:nn{ijitcs}{
	volume.tl_set:N = \l__ijitcs_volume_tl,
	month.tl_set:N = \l__ijitcs_month_tl,
	year.tl_set:N = \l__ijitcs_year_tl,
	startpage.tl_set:N = \l__ijitcs_startpage_tl,
	endpage.tl_set:N = \l__ijitcs_endpage_tl,
	doi.tl_set:N = \l__ijitcs_doi_tl,
	title.tl_set:N = \l__ijitcs_title_tl,
	abstract.tl_set:N = \l__ijitcs_abstract_tl,
	keywords.tl_set:N = \l__ijitcs_keywords_tl,
	received.tl_set:N = \l__ijitcs_received_tl,
	issue.tl_set:N = \l__ijitcs_issue_tl
}

\usepackage{fancyhdr}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0pt}
\fancyhf{}
% \fancyhead[RE, LO]{ }
\fancyhead[C]{{\fontsize{9pt}{9pt} \selectfont \tl_use:N \l__ijitcs_title_tl}}
% \fancyhead[RO,LE]{{\fontsize{9pt}{9pt} \selectfont\bfseries \thepage}}

\fancyfoot[LE,RO]{\fontsize{9pt}{9pt} \selectfont\bfseries \thepage}
\fancyfoot[RE,LO]{\fontsize{9pt}{9pt} Volume~\tl_use:N \l__ijitcs_volume_tl{}~(\tl_use:N \l__ijitcs_year_tl),~Issue~\tl_use:N \l__ijitcs_issue_tl}


\fancypagestyle{thefirstpage}{
	\fancyhead{}
	\fancyhead[LO]{{\fontsize{10.5pt}{10.5pt} \selectfont \textbf{\textit{I.~J.~Information~Technology~and~Computer~Science}}, ~\textbf{\tl_use:N \l__ijitcs_year_tl,~\tl_use:N \l__ijitcs_issue_tl,~\tl_use:N \l__ijitcs_startpage_tl-\tl_use:N \l__ijitcs_endpage_tl}}\\
		{\fontsize{10}{10} \selectfont ~Published~Online~on~ \tl_use:N \l__ijitcs_month_tl, {}~\tl_use:N \l__ijitcs_year_tl {}~by ~MECS~Press~(http://www.mecs-press.org/)} \\ DOI:~\tl_use:N \l__ijitcs_doi_tl }
	\fancyhead[CO]{ }
	\fancyhead[RO]{\includegraphics[height = 1.6cm]{logo.png}}
	\fancyfoot[L]{\fontsize{9pt}{9pt}\selectfont This~work~is~open~access~and~licensed~under~the~Creative~Commons~CC~BY~4.0~License.}
	\fancyfoot[R]{\fontsize{9pt}{9pt}\selectfont Volume~\tl_use:N \l__ijitcs_volume_tl{}~(\tl_use:N \l__ijitcs_year_tl),~Issue~\tl_use:N \l__ijitcs_issue_tl}
}

\NewDocumentCommand{\IjitcsSet}{m}{\keys_set:nn{ijitcs}{#1}}


\clist_new:N \l_ijitcs_authors_list 

\RenewDocumentCommand{\author}{s+m+m+m+m}{
	\IfBooleanTF{#1}{
		\clist_put_right:Nn \l_ijitcs_authors_list {
			\begin{flushleft}
				{\fontsize{11pt}{11pt} \selectfont\bfseries #2*}\\
				{\fontsize{10pt}{10pt} \selectfont #3} \\
				{E-mail:~\fontsize{10pt}{10pt} \selectfont #4} \\
				{ORCID iD:~\fontsize{10pt}{10pt} \selectfont #5} \\
				{*Corresponding Author} 
			\end{flushleft}
		}
	}{
		\clist_put_right:Nn \l_ijitcs_authors_list {
			\begin{flushleft}
				{\fontsize{11pt}{11pt} \selectfont\bfseries #2}\\
				{\fontsize{10pt}{10pt} \selectfont #3} \\
				{E-mail:~\fontsize{10pt}{10pt} \selectfont #4} \\
				{ORCID iD:~\fontsize{10pt}{10pt} \selectfont #5} 
			\end{flushleft}
		}
	}
}


\usepackage{ragged2e}


%% title 
\renewcommand{\maketitle}{
	\vspace*{-2pt}
	{\justifying\noindent\fontsize{24}{24} \selectfont \tl_use:N \l__ijitcs_title_tl}
	\begin{flushleft}%
	\medskip	
	\vspace{0.4cm} \par
	\clist_use:Nn \l_ijitcs_authors_list {\vspace*{-4pt}}
	\vspace*{.2cm}
	\end{flushleft}
	Received:~\tl_use:N \l__ijitcs_received_tl\\[.8cm]

	\noindent {\fontsize{10pt}{10pt} \selectfont {\bfseries{Abstract:}}~\tl_use:N \l__ijitcs_abstract_tl} \par
	\vspace{0.3cm}
	\noindent {\fontsize{10pt}{10pt} \selectfont {\bfseries{Index Terms:}}~\tl_use:N \l__ijitcs_keywords_tl} \par
	\vspace*{.5cm}
	
}
\ExplSyntaxOff




\usepackage{graphicx}


%% section 
\usepackage{indentfirst}
\usepackage{titlesec}
\titleformat{\section}[hang]{\bfseries\fontsize{11pt}{11pt}\selectfont}{\arabic{section}.}{.75ex}{}
\titleformat{\subsection}[hang]{\itshape\fontsize{10pt}{10pt}\selectfont}{\arabic{section}.\arabic{subsection}.}{.75ex}{}

\titlespacing*{\section}{0pt}{12pt}{12pt}
\titlespacing*{\subsection}{0pt}{6pt}{6pt}



%% table and figure

\usepackage{caption}
\DeclareCaptionFont{eight}{\fontsize{8pt}{8pt}\selectfont}
\captionsetup{singlelinecheck = false,font=eight,justification=raggedright,labelsep = period}

\captionsetup[figure]{name=Fig}
\captionsetup[table]{name=Table}



\usepackage{wrapstuff}

	

\newenvironment{AuthorsProfiles}{%
\begin{flushleft}
	\textbf{Authors' Profiles}
\end{flushleft}
\fontsize{9pt}{9pt}\selectfont
}{}
\newcommand{\authoritem}[2]{%
\begin{wrapstuff}[l,top=0]
    \includegraphics[width = 2.5cm]{#1}
\end{wrapstuff}
#2%
\wrapstuffclear
\vspace*{1cm}
}
\endinput