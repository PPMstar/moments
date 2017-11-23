c
#define nndumps 480
#define nnimage1 5
#define nnimage2 8
#define iincimage 1
#define isdecompress 0
#define isBoB8 0
#define isBoB 0
#define isMom 0
#define isMomAll 0*isMom
#define isvort 0
#define isdivu 0
#define isuy 0
#define isenuc 0
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#define nnxteams 2
#define nnzteams 2
#define nnyteams 2
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
#define nntzbricks 8
#define nntxbricks nntzbricks
#define nntybricks 4
#define nnnnx 48
#define nnnny 96
#define nnnnz nnnnx
c
#define nntybricksremove 0
#define nnewZteamsperOldZteam 2
c
#define isintelfortran 1
c
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c     This version will build phoney team files that bob2hv is able to
c     handle.  Otherwise, we run into the fact that the index variables
c     used in that utility are 32-bit integers.
c     For simplicity, we chop up the brick in the Z dimension after we
c     have composed it in memory.
c     We do this trick only for the largest output, the doubly resolved
c     fractional volume data (BoB8).
c     We also do this trick only for the special case where the input
c     consists of only a single team.
c
c     The flag  nntybricksremove  allows us to obtain a trivial sort of
c     data compression by leaving off some of the data that, for TP1,
c     contains nothing of significance.
c
      program runit
c
c
      integer*8   NTXBricks,NTYBricks,NTZBricks,NTBricks
      integer*8   NXTeams,NYTeams,NZTeams,NTeams
      integer*8   NXBricks,NYBricks,NZBricks,NBricks
      integer*8   nx,ny,nz,ntybricksremove
c
      parameter (NTZBricks=nntzbricks)
      parameter (NTXBricks=nntxbricks)
      parameter (NTYBricks=nntybricks)
      parameter (NTBricks=NTXBricks*NTYBricks*NTZBricks)
      parameter (NZTeams=nnzteams)
      parameter (NXTeams=nnxteams)
      parameter (NYTeams=nnyteams)
      parameter (NTeams=NXTeams*NYTeams*NZTeams)
      parameter (NZBricks=NZTeams*NTZBricks)
      parameter (NXBricks=NXTeams*NTXBricks)
      parameter (NYBricks=NYTeams*NTYBricks)
      parameter (NBricks=NXBricks*NYBricks*NZBricks)
c
      parameter (nfluids=1)
      parameter (nx=nnnnx)
      parameter (ny=nnnny)
      parameter (nz=nnnnz)
      parameter (ntybricksremove=nntybricksremove)
      parameter (newZteamsperOldZteam=nnewZteamsperOldZteam)
      parameter (newzteams=newZteamsperOldZteam*NZTeams)
c
c
      integer*4   BoBbricks((nx*ny*nz/4)*NTBricks/newZteamsperOldZteam)
      common / BoBscratch1 / BoBbricks
#if isBoB8
      integer*4   BoB8bricks((8*nx*ny*nz/4)*NTBricks
     &                                      /newZteamsperOldZteam)
      common / BoB8scratch1 / BoB8bricks
#endif
      character*1   momsbricks((nx*ny*nz/64)*48*NTBricks)
      common / momsscratch1 / momsbricks
c
      integer*4   BoB(nx*NTXBricks*ny*(NTYBricks-2*ntybricksremove)
     &                  *nz*NTZBricks/(4*newZteamsperOldZteam))
      common / BoBscratch2 / BoB
#if isBoB8
      integer*4   BoB8(8*nx*NTXBricks*ny*(NTYBricks-2*ntybricksremove)
     &                  *nz*NTZBricks/(4*newZteamsperOldZteam))
      common / BoB8scratch2 / BoB8
#endif
      character*1   moms((nx*ny*nz/64)*48*NTXBricks
     &                  *(NTYBricks-2*ntybricksremove)*NTZBricks)
      common / momsscratch2 / moms
      character*1   moms48((nx*ny*nz/64)*NTXBricks
     &                    *(NTYBricks-2*ntybricksremove)*NTZBricks,48)
      equivalence  (moms48,moms)
      character*1   mom((nx*ny*nz/64)*NTXBricks
     &                 *(NTYBricks-2*ntybricksremove)*NTZBricks)
      common / momsscratch3 / mom
      integer*8   lenBoB8, lenBoB
c
#if isdecompress
      dimension   GammaCld(nfluids), RhoCldtoAir0(nfluids)
      dimension   ArgsImg(18)
      dimension   PerfDataTeam(128), PerfBrick(128,NTBricks)
      real*8   TotalPerf(9), totalsecs
      dimension   aaa(512), iaaa(512)
      real*8   aaa8(256)
      equivalence (iaaa,aaa)
      equivalence (aaa8,aaa)
      dimension   fv(2*nx*NTXBricks,2*ny*(NTYBricks-2*ntybricksremove),
     &                              2*nz*NTZBricks)
      character*1  afv(2*nx*NTXBricks,
     &                 2*ny*(NTYBricks-2*ntybricksremove),
     &                 2*nz*NTZBricks/newZteamsperOldZteam)
      equivalence   (afv,BoB8)
#endif
c
c
      character*20   filemoms48, filehires8, filevort, fileuy
      character*20   gilemoms48, gilehires8, gilevort, gileuy
      character*20   hilemoms48, hilehires8, hilevort, hileuy
      character*20   filep, filerho, filedivu, fileenuc
      character*20   gilep, gilerho, giledivu, gileenuc
      character*20   hilep, hilerho, hiledivu, hileenuc
c     character*45   thefilename
      character*23   thefilename
      character*1 suffix1, suffix2, suffix3, zuffix3
      character*100 argv
c
      filemoms48 = 'FVandMoms48-0000.bob'
      filehires8 = 'FV-hires01-0000.bob8'
      filevort = 'Lg10Vort-01-0000.bob'
      fileuy = 'TanhUY--001-0000.bob'
      filep = 'Lg10P-SL-01-0000.bob'
      filerho = 'Lg10Rho-001-0000.bob'
      filedivu = 'TanhDivU-01-0000.bob'
      fileenuc = 'Lg10ENUCbyP-0000.bob'
c
      gilemoms48 = 'FVandMomt48-0000.bob'
      gilehires8 = 'FV-hiret01-0000.bob8'
      gilevort = 'Lg10Voru-01-0000.bob'
      gileuy = 'TanhUY-0001-0000.bob'
      gilep = 'Lg10P-SM-01-0000.bob'
      gilerho = 'Lg10Rhp-001-0000.bob'
      giledivu = 'TanhDivV-01-0000.bob'
      gileenuc = 'Lg10ENVCbyP-0000.bob'
c
      hilemoms48 = 'FVandMomu48-0000.bob'
      hilehires8 = 'FV-hireu01-0000.bob8'
      hilevort = 'Lg10Vorv-01-0000.bob'
      hileuy = 'TanhUY-1001-0000.bob'
      hilep = 'Lg10P-SN-01-0000.bob'
      hilerho = 'Lg10Rhq-001-0000.bob'
      hiledivu = 'TanhDivW-01-0000.bob'
      hileenuc = 'Lg10ENWCbyP-0000.bob'
c
      ifBoB8 = isBoB8
      ifBoB = isBoB
      ifMom = isMom
      ifMomAll = isMomAll*ifMom

      nimage1 = 0 
      call getarg(1,argv)
      do i=1,LEN_TRIM(argv)
          nimage1 = nimage1*10 + (ichar(argv(i:i)) - ichar('0'))
      enddo

      nimage2 = 0 
      call getarg(2,argv)
      do i=1,LEN_TRIM(argv)
          nimage2 = nimage2*10 + (ichar(argv(i:i)) - ichar('0'))
      enddo
c
      lenmom = (nx*ny*nz/64)*NTXBricks
     &                      *(NTYBricks-2*ntybricksremove)*NTZBricks
      lenBoB8 = 8*nx*NTXBricks*ny*(NTYBricks-2*ntybricksremove)
     &              *nz*NTZBricks/(4*newZteamsperOldZteam)
      lenBoB = lenBoB8 / 8
c
c
#if isdecompress
c
c     Read in the parameters of the run from the tiny component of the
c     restart dump.
c
      open (36,file='restartaaa',form='binary',
     &      access='sequential',position='rewind',status='unknown')
      read (36) aaa
      close (36)
c
      g = aaa(1)
      TransitionThickness = aaa(2)
      DisplaceAmpl = aaa(3)
      GammaAir = aaa(4)
      GammaCld(1) = aaa(5)
      RhoCldToAir0(1) = aaa(6)
      BrickWidth = aaa(7)
      BrickHeight = aaa(8)
      BrickDepth = aaa(9)
      rho00 = aaa(10)
      p00 = aaa(11)
      VX0 = aaa(12)
      VX0Air = aaa(13)
      VX0Cld = aaa(14)
      VCirculationAir = aaa(15)
      VCirculationCld = aaa(16)
      Ampl1 = aaa(17)
      Ampl11 = aaa(18)
      VY1 = aaa(19)
      VY11 = aaa(20)
      asymm = aaa(21)
      SoundSpeed0Air = aaa(22)
      SoundSpeed0Cld = aaa(23)
      ux0 = aaa(24)
      ux1 = aaa(25)
      uy0 = aaa(26)
      uy1 = aaa(27)
      uz0 = aaa(28)
      uz1 = aaa(29)
      vx0sq = aaa(30)
      rho0 = aaa(31)
      rho1 = aaa(32)
      p0 = aaa(33)
      p1 = aaa(34)
      s00 = aaa(35)
      s01 = aaa(36)
      s0 = aaa(37)
      s1 = aaa(38)
      fv0 = aaa(39)
      fv1 = aaa(40)
      vort0 = aaa(41)
      vort1 = aaa(42)
      divu0 = aaa(43)
      divu1 = aaa(44)
      deex = aaa(45)
      dtinit = aaa(46)
      safety = aaa(47)
      difcon = aaa(48)
      shkjmp = aaa(49)
      wavhaf = aaa(50)
      do i = 1,18
         argsimg(i) = aaa(50+i)
      enddo
      smlrho = aaa(69)
      smallp = aaa(70)
      smalle = aaa(71)
      smallu = aaa(72)
      plotsperperiod = aaa(73)
      periods = aaa(74)
      tstop = aaa(75)
      dtdump = aaa(76)
      do i = 1,max(20,nfluids)
         GammaCld(i) = aaa(80+i)
      enddo
      do i = 1,max(20,nfluids)
         RhoCldToAir0(i) = aaa(100+i)
      enddo
#endif
c
c
c     We want to read in the data incrementally, and to write it out
c     also incrementally.  We take the requested number of Z-teams and
c     from that compute how large each of the Z-team files will be.
c     We will demand that each Z-team file be an integer number of
c     Z-nodes thick.  If this does not fit the requested number of such
c     files, then we will flag an error and quit.
c     We will build up each new Z-team file in memory, and then write
c     it out to disk all at once.
c     We will apply one additional constraint, namely that an integer
c     number of our new Z-teams fit into each of our old Z-teams.
c
c     The way this will work is that we will leave the numbers of
c     X- and Y-teams unchanged, but we will multiply the number of
c     Z-teams by the factor  newZteamsperOldZteam.
c
         NNZBricks = 2
         ntznodes = NTZBricks / NNZBricks
         if (NNZBricks*ntznodes .ne. NTZBricks)   NNZBricks = 1
         ntznodes = NTZBricks / NNZBricks
         nodesperZteam = NTZBricks * NZTeams / (newzteams * NNZBricks)
         if ((nodesperZteam * newzteams)
     &       .ne. (ntznodes * NZTeams))   then
            write (6,*) 'The number of requested new Z teams does not'
            write (6,*) 'result in an integer number of Z-nodes per'
            write (6,*) 'new Z-team.   Job Killed!'
            write (6,*) 'ZnodesperZteam, newzteams =',
     &                   ZnodesperZteam, newzteams
            write (6,*) 'ZnodesperZteam, newzteams =',ntznodes,NZTeams
            STOP
         endif
c        newZteamsperOldZteam = newzteams / NZTeams
c
c
      do nimage = nimage1,nimage2,iincimage
      numb1000 = nimage / 1000
      numb100 = (nimage / 100)  -  10 * numb1000
      numb10 = (nimage / 10)  -  100 * numb1000  -  10 * numb100
      numb1 = nimage  -  1000 * numb1000  -  100 * numb100
     &                                    -  10 * numb10
c
      do myteamz = 0,NZTeams-1
c      do myteamz = 1,1
      do myteamy = 0,NYTeams-1
      do myteamx = 0,NXTeams-1
c
      i = ichar('a')
      suffix1 = achar(i+myteamx)
      suffix2 = achar(i+myteamy)
      suffix3 = achar(i+myteamz)
#if isBoB8
      if (ifBoB8 .gt. 0)   then
         write(thefilename,667)  filehires8,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         write (6,*) thefilename
         open (37,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
      endif
#endif
      if (ifBoB .gt. 0)   then
#if isvort
         write(thefilename,666)  filevort,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         open (38,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
#endif
#if isuy
         write(thefilename,666)  fileuy,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         open (39,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
#endif
#if isdivu
         write(thefilename,666)  filedivu,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         open (40,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
#endif
#if isenuc
         write(thefilename,666)  fileenuc,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         open (41,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
#endif
c         write(thefilename,666)  filerho,numb1000,numb100,
c     &                           numb10,numb1,suffix1,suffix2,suffix3
c         open (41,file=thefilename,form='binary',
c     &         access='sequential',position='rewind',status='unknown')
c         write(thefilename,666)  filep,numb1000,numb100,
c     &                           numb10,numb1,suffix1,suffix2,suffix3
c         open (42,file=thefilename,form='binary',
c     &         access='sequential',position='rewind',status='unknown')
      endif
c
      do mynewsubteamz = 0,newZteamsperOldZteam-1
      i = ichar('a')
      suffix1 = achar(i+myteamx)
      suffix2 = achar(i+myteamy)
      suffix3 = achar(i+(myteamz*newZteamsperOldZteam)+mynewsubteamz)
c      
#if isBoB8
      if (ifBoB8 .gt. 0)   then
c
         read (37) BoB8bricks
         call reformatBoB8(BoB8bricks,BoB8,nx,ny,nz,ntybricksremove,
     &                     NTXBricks,NTYBricks,
     &                     NTZBricks/newZteamsperOldZteam)
#if isdecompress
c
c     Uncompress the fractional volume data.
c     The rescaling of this variable is nonlinear and therefore we must
c     use Newton's method to invert it.
c
         fv000 = ArgsImg(7)
         fv001 = ArgsImg(8)
         fv001inv = 1. / fv001
         fvtest = fv000 + 1. - fv001
         fvtest = fvtest - .0000001
c
c        fvtest is negative when  fv000 = 0  and  fv001 = 1.
c        In this case, we will not rescale  fv  at all.
c
         deenom = 1. / 245.499
c
         do k8 = 1,2*nz*NTZBricks
         do j8 = 1,2*ny*(NTYBricks-2*ntybricksremove)
         do i8 = 1,2*nx*NTXBricks
            ifv = iachar(afv(i8,j8,k8))
            effv = ifv
            effv = (effv - 5.) * deenom
            saveit = effv
            thyng = 2. * effv  -  1.
            if (fv000 .gt. 0.)   effv = thyng
c
c     We must now invert the following nonlinear transformation.
c     We will assume that  fv000 = 0   and that   0 < fv001 < 1.
c
c           effv = (thyng - 1.) / (thyng + 1.)
c     where         thyng = (a + b)**2
c     where             a = (fv - fv000) * fv001inv * rhocldtoair(1)
c     and               b = sqrt (a**2 + 1.)
c
c     The original scaling, performed in the code, is repeated here:
c
c               saveit = fvlbnimg(j,m)
c               thyng = (fvlbnimg(j,m) - fv000) * fv001inv
c               thyng = thyng * rhocldtoair0(m)
c               thang = thyng * thyng  +  1.
c               thang = sqrt(thang)
c               thyng = thyng + thang
c               thyng = thyng * thyng
c               thang = thyng + 1.
c               thyng = (thyng - 1.) / thang
c               fvlbnimg(j,m) = thyng
c               thang = (thyng + 1.) * .5
c               if (fv000 .gt. 0.)   fvlbnimg(j,m) = thang
cc                    fvtest = (fv000 + 1. - fv001)  -  .0000001
c               if (fvtest .lt. 0.)   fvlbnimg(j,m) = saveit
c               fvlbnimg(j,m) = 245.499 * fvlbnimg(j,m)  +  5.
c
            fv(i8,j8,k8) = effv
         enddo
         enddo
         enddo
#endif
         if (ntybricksremove .eq. 0)   then
            write(thefilename,667)  gilehires8,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         else
            write(thefilename,667)  hilehires8,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         endif
         write (6,*) thefilename
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) BoB8
         close (36)
      endif
#endif
      if (ifBoB .gt. 0)   then
#if isvort
         read (38) BoBbricks
         call reformatBoB(BoBbricks,BoB,nx,ny,nz,ntybricksremove,
     &                    NTXBricks,NTYBricks,
     &                    NTZBricks/newZteamsperOldZteam)
         if (ntybricksremove .eq. 0)   then
            write(thefilename,666)  gilevort,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         else
            write(thefilename,666)  hilevort,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         endif
         write (6,*) thefilename
         open (36,file=thefilename,form='binary',
     &      access='sequential',position='rewind',status='unknown')
         write (36) BoB
         close (36)
#endif
c
#if isuy
         read (39) BoBbricks
         call reformatBoB(BoBbricks,BoB,nx,ny,nz,ntybricksremove,
     &                    NTXBricks,NTYBricks,
     &                    NTZBricks/newZteamsperOldZteam)
         if (ntybricksremove .eq. 0)   then
            write(thefilename,666)  gileuy,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         else
            write(thefilename,666)  hileuy,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         endif
         open (36,file=thefilename,form='binary',
     &      access='sequential',position='rewind',status='unknown')
         write (36) BoB
         close (36)
#endif
c
#if isdivu
         read (40) BoBbricks
         call reformatBoB(BoBbricks,BoB,nx,ny,nz,ntybricksremove,
     &                    NTXBricks,NTYBricks,
     &                    NTZBricks/newZteamsperOldZteam)
         if (ntybricksremove .eq. 0)   then
            write(thefilename,666)  giledivu,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         else
            write(thefilename,666)  hiledivu,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         endif
         open (36,file=thefilename,form='binary',
     &      access='sequential',position='rewind',status='unknown')
         write (36) BoB
         close (36)
#endif
c
#if isenuc
         read (41) BoBbricks
         call reformatBoB(BoBbricks,BoB,nx,ny,nz,ntybricksremove,
     &                    NTXBricks,NTYBricks,
     &                    NTZBricks/newZteamsperOldZteam)
         if (ntybricksremove .eq. 0)   then
            write(thefilename,666)  gileenuc,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         else
            write(thefilename,666)  hileenuc,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         endif
         open (36,file=thefilename,form='binary',
     &      access='sequential',position='rewind',status='unknown')
         write (36) BoB
         close (36)
#endif
      go to 12341
c
         read (41) BoBbricks
         call reformatBoB(BoBbricks,BoB,nx,ny,nz,ntybricksremove,
     &                    NTXBricks,NTYBricks,
     &                    NTZBricks/newZteamsperOldZteam)
         if (ntybricksremove .eq. 0)   then
            write(thefilename,666)  gilerho,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         else
            write(thefilename,666)  hilerho,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         endif
         open (36,file=thefilename,form='binary',
     &      access='sequential',position='rewind',status='unknown')
         write (36) BoB
         close (36)
c
         read (42) BoBbricks
         call reformatBoB(BoBbricks,BoB,nx,ny,nz,ntybricksremove,
     &                    NTXBricks,NTYBricks,
     &                    NTZBricks/newZteamsperOldZteam)
         if (ntybricksremove .eq. 0)   then
            write(thefilename,666)  gilep,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         else
            write(thefilename,666)  hilep,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         endif
         open (36,file=thefilename,form='binary',
     &      access='sequential',position='rewind',status='unknown')
         write (36) BoB
         close (36)
12341 continue
      endif
c
      enddo
c               This is the end of the loop on  mynewsubteamz
#if isBoB8
      if (ifBoB8 .gt. 0)   then
         close (37)
      endif
#endif
      if (ifBoB .gt. 0)   then
#if isvort
         close (38)
#endif
#if isuy
         close (39)
#endif
#if isdivu
         close (40)
#endif
#if isenuc
         close (41)
#endif
c         close (41)
c         close (42)
      endif
      i = ichar('a')
      suffix1 = achar(i+myteamx)
      suffix2 = achar(i+myteamy)
      suffix3 = achar(i+myteamz)
c
      if (ifMom .gt. 0)   then
         write(thefilename,666)  filemoms48,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         read (36) momsbricks
         close (36)
         call reformatmoms(momsbricks,moms,nx/2,ny/2,nz/2,
     &                     ntybricksremove,
     &                     NTXBricks,NTYBricks,NTZBricks)
         if (ntybricksremove .eq. 0)   then
            write(thefilename,666)  gilemoms48,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         else
            write(thefilename,666)  hilemoms48,numb1000,numb100,
     &                           numb10,numb1,suffix1,suffix2,suffix3
         endif
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) moms
         close (36)
      endif
c
      if (ifMomAll .gt. 0)   then
c      filemoms48 = 'FVandMoms48-0000.bob'
         do imom = 1,lenmom
            mom(imom) = moms48(imom,17)
         enddo
      write(thefilename,666)  'Rho-Lores01-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,19)
         enddo
      write(thefilename,666)  'P---Lores01-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,21)
         enddo
      write(thefilename,666)  'Vorticity-A-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,23)
         enddo
      write(thefilename,666)  'RhoUX-Lores-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,25)
         enddo
      write(thefilename,666)  'RhoUY-Lores-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,27)
         enddo
      write(thefilename,666)  'RhoUZ-Lores-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,29)
         enddo
      write(thefilename,666)  'RhoUXUX-Avg-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,31)
         enddo
      write(thefilename,666)  'RhoUXUY-Avg-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,33)
         enddo
      write(thefilename,666)  'RhoUXUZ-Avg-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,35)
         enddo
      write(thefilename,666)  'RhoUYUY-Avg-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,37)
         enddo
      write(thefilename,666)  'RhoUYUZ-Avg-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,39)
         enddo
      write(thefilename,666)  'RhoUZUZ-Avg-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,41)
         enddo
      write(thefilename,666)  'RhoHUX-Avgd-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,43)
         enddo
      write(thefilename,666)  'RhoHUY-Avgd-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,45)
         enddo
      write(thefilename,666)  'RhoHUZ-Avgd-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
         do imom = 1,lenmom
            mom(imom) = moms48(imom,47)
         enddo
      write(thefilename,666)  'NegDivU-Avg-',numb1000,numb100,
     &                        numb10,numb1,suffix1,suffix2,suffix3
         open (36,file=thefilename,form='binary',
     &         access='sequential',position='rewind',status='unknown')
         write (36) mom
         close (36)
      endif
c
666   format (a12,4i1,'.bob',3a1)
667   format (a11,4i1,'.bob8',3a1)
c
      enddo
      enddo
      enddo
c
      enddo
c
c
      stop
      end
c
c
c
c=======================================================================
c
c
c
c=======================================================================
c
c
c
      subroutine reformatBoB(iBoBbrick,iBoB,nx,ny,nz,
     &                       ntybricksremove,
     &                       NTXBricks,NTYBricks,NTZBricks)
c
      integer*8   NTXBricks,NTYBricks,NTZBricks
      integer*8   nx,ny,nz,ntybricksremove
c
c     integer*4   iBoBbrick(nx/4,ny,nz,9,NTXBricks*NTYBricks*NTZBricks)
      integer*4  iBoBbrick((nx/4)*ny*nz,NTXBricks*NTYBricks*NTZBricks)
      integer*4  iBoB(nx*NTXBricks/4,ny*(NTYBricks-2*ntybricksremove),
     &                               nz*NTZBricks)
c
      integer*8   nxby4,nxby2,nynxby4,ninbrick,mybrick,mybb
      integer*8   kn,knb,kb,koff,jn,jnb,jb,joff,in,inb,ib,ioff
      integer*8   kk,k,kkoff,jj,j,jjkkoff,ii,i,iijjkk
      integer*8   NNXBricks,NNYBricks,NNZBricks,NNBricks
      integer*8   ntxnodes,ntynodes,ntznodes,MyTNode
c
c      write (6,*) 'nx,ny,nz,nfluids,NXBricks,NYBricks,NZBricks =',
c     &             nx,ny,nz,nfluids,NXBricks,NYBricks,NZBricks
      nxby4 = nx / 4
      if (nx .ne. 4*nxby4)   then
      open (37,file='ErrorReformatBoB.ppm',form='formatted',
     &         access='sequential',position='append',status='unknown')
      write (37,3737)   nx
3737  format (' Error in ReformatBoB.  Nx is not a multiple of 4.'/
     &        ' nx = ',i6/' Job Killed.'/)
      STOP
      close (37)
      endif
c
c     We have teams inside a domain, nodes in side a team, and bricks
c     inside a team.  This gives us the right assignments of MPI ranks
c     (bricks) to network nodes.  There is no advantage to doing this
c     careful assignment unless NTXBricks is even, and similarly for
c     Y and Z.  So we do it only in that case.
c
      NNXBricks = 2
      ntxnodes = NTXBricks / NNXBricks
      if (NNXBricks*ntxnodes .ne. NTXBricks)   NNXBricks = 1
      ntxnodes = NTXBricks / NNXBricks
      NNYBricks = 2
      ntynodes = NTYBricks / NNYBricks
      if (NNYBricks*ntynodes .ne. NTYBricks)   NNYBricks = 1
      ntynodes = NTYBricks / NNYBricks
      NNZBricks = 2
      ntznodes = NTZBricks / NNZBricks
      if (NNZBricks*ntznodes .ne. NTZBricks)   NNZBricks = 1
      ntznodes = NTZBricks / NNZBricks
      NNBricks = NNXBricks * NNYBricks * NNZBricks
c
c      MyTBrick = myrank - myteam*ntbricks
c
c      mytnode = MyTBrick / NNBricks
c      mytnodeZ = mytnode / (ntxnodes * ntynodes)
c      mytnodeX = mytnode  -  mytnodeZ * ntxnodes * ntynodes
c      mytnodeY = mytnodeX / ntxnodes 
c      mytnodeX = mytnodeX  -  mytnodeY * ntxnodes
c
c     The following coordinates are 1-based rather than 0-based.
c
c      MyTNBrick = MyTBrick  -  mytnode * NNBricks
c      MyTNBrickZ = MyTNBrick  / (NNXBricks * NNYBricks)
c      MyTNBrickX = MyTNBrick  -  MyTNBrickZ * NNXBricks * NNYBricks
c      MyTNBrickZ = MyTNBrickZ + 1
c      MyTNBrickY = MyTNBrickX / NNXBricks
c      MyTNBrickX = (MyTNBrickX + 1) -  MyTNBrickY * NNXBricks
c      MyTNBrickY = MyTNBrickY + 1
c
c      MyBrick = (MyTNBrickX-1) +  NNXBricks * (MyTNBrickY - 1
c     &                         +  NNYBricks * (MyTNBrickZ - 1))
c     &                         +  MyTNode * NNBricks
c
c     The above is a 0-based numbering for the bricks, like the MPI
c     ranks.  In the code below, we need a 1-based numbering.
c
      nxby2 = 2 * nxby4
      nynxby4 = ny * nxby4
      ninbrick = nz * nynxby4
c
      do kn = 1,ntznodes
      do knb = 1,NNZBricks
         kb = (kn - 1 ) * NNZBricks  +  knb
         koff = (kb - 1) * nz
         do jn = 1,ntynodes
         do jnb = 1,NNYBricks
            jb = (jn - 1) * NNYBricks  +  jnb
            if ((jb .gt. ntybricksremove) .and.
     &          (jb .le. NTYBricks-ntybricksremove))   then
            joff = (jb-ntybricksremove - 1) * ny
            do in = 1,ntxnodes
               MyTNode = (in - 1)  +  ntxnodes * (jn - 1
     &                             +  ntynodes * (kn - 1))
               mybb = MyTNode * NNBricks
               mybb = mybb  +  NNXBricks * (jnb - 1
     &                      +  NNYBricks * (knb - 1))
            do inb = 1,NNXBricks
               ib = (in - 1) * NNXBricks  +  inb
               ioff = (ib - 1) * nxby4
               mybrick = mybb + inb
               do kk = 1,nz
                  k = koff + kk
                  kkoff = (kk - 1) * nynxby4
                  do jj = 1,ny
                     j = joff + jj
                     jjkkoff = kkoff  +  (jj - 1) * nxby4
                     do ii = 1,nxby4
                     i = ioff + ii
                     iijjkk = ii + jjkkoff
                     iBoB(i,j,k) = iBoBbrick(iijjkk,mybrick)
                     enddo
                  enddo
               enddo
            enddo
            enddo
            endif
         enddo
         enddo
      enddo
      enddo
c
      return
      end
c
c
c
c=======================================================================
c
c
c
c=======================================================================
c
c
c
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#if isBoB8
      subroutine reformatBoB8(iBoB8brick,iBoB8,nx,ny,nz,
     &                        ntybricksremove,
     &                        NTXBricks,NTYBricks,NTZBricks)
c
      integer*8   NTXBricks,NTYBricks,NTZBricks
      integer*8   nx,ny,nz,ntybricksremove
c
      integer*4  iBoB8brick(2*nx*ny*nz,NTXBricks*NTYBricks*NTZBricks)
      integer*4  iBoB8(nx*NTXBricks/2,
     &                 2*ny*(NTYBricks-2*ntybricksremove),
     &                 2*nz*NTZBricks)
c
      integer*8   nxby4,nxby2,nynxby4,ninbrick,mybrick,mybb
      integer*8   kn,knb,kb,koff,koff2,jn,jnb,jb,joff,joff2
      integer*8   in,inb,ib,ioff,ioff2
      integer*8   kk,k,jj,j,ii,i,ijk
      integer*8   NNXBricks,NNYBricks,NNZBricks,NNBricks
      integer*8   ntxnodes,ntynodes,ntznodes,MyTNode
c
      nxby4 = nx / 4
      if (nx .ne. 4*nxby4)   then
      open (37,file='ErrorReformatBoB8.ppm',form='formatted',
     &         access='sequential',position='append',status='unknown')
      write (37,3737)   nx
3737  format (' Error in ReformatBoB8.  Nx is not a multiple of 4.'/
     &        ' nx = ',i6/' Job Killed.'/)
      STOP
      close (37)
      endif
c
c     We have teams inside a domain, nodes in side a team, and bricks
c     inside a team.  This gives us the right assignments of MPI ranks
c     (bricks) to network nodes.  There is no advantage to doing this
c     careful assignment unless NTXBricks is even, and similarly for
c     Y and Z.  So we do it only in that case.
c
      NNXBricks = 2
      ntxnodes = NTXBricks / NNXBricks
      if (NNXBricks*ntxnodes .ne. NTXBricks)   NNXBricks = 1
      ntxnodes = NTXBricks / NNXBricks
      NNYBricks = 2
      ntynodes = NTYBricks / NNYBricks
      if (NNYBricks*ntynodes .ne. NTYBricks)   NNYBricks = 1
      ntynodes = NTYBricks / NNYBricks
      NNZBricks = 2
      ntznodes = NTZBricks / NNZBricks
      if (NNZBricks*ntznodes .ne. NTZBricks)   NNZBricks = 1
      ntznodes = NTZBricks / NNZBricks
      NNBricks = NNXBricks * NNYBricks * NNZBricks
c
c      MyTBrick = myrank - myteam*ntbricks
c
c      mytnode = MyTBrick / NNBricks
c      mytnodeZ = mytnode / (ntxnodes * ntynodes)
c      mytnodeX = mytnode  -  mytnodeZ * ntxnodes * ntynodes
c      mytnodeY = mytnodeX / ntxnodes 
c      mytnodeX = mytnodeX  -  mytnodeY * ntxnodes
c
c     The following coordinates are 1-based rather than 0-based.
c
c      MyTNBrick = MyTBrick  -  mytnode * NNBricks
c      MyTNBrickZ = MyTNBrick  / (NNXBricks * NNYBricks)
c      MyTNBrickX = MyTNBrick  -  MyTNBrickZ * NNXBricks * NNYBricks
c      MyTNBrickZ = MyTNBrickZ + 1
c      MyTNBrickY = MyTNBrickX / NNXBricks
c      MyTNBrickX = (MyTNBrickX + 1) -  MyTNBrickY * NNXBricks
c      MyTNBrickY = MyTNBrickY + 1
c
c      MyBrick = (MyTNBrickX-1) +  NNXBricks * (MyTNBrickY - 1
c     &                         +  NNYBricks * (MyTNBrickZ - 1))
c     &                         +  MyTNode * NNBricks
c
c     The above is a 0-based numbering for the bricks, like the MPI
c     ranks.  In the code below, we need a 1-based numbering.
c
      nxby2 = 2 * nxby4
      nynxby4 = ny * nxby4
      ninbrick = nz * nynxby4
c
      do kn = 1,ntznodes
      do knb = 1,NNZBricks
         kb = (kn - 1 ) * NNZBricks  +  knb
         koff = (kb - 1) * nz
         koff2 = 2 * koff
         do jn = 1,ntynodes
         do jnb = 1,NNYBricks
            jb = (jn - 1) * NNYBricks  +  jnb
            if ((jb .gt. ntybricksremove) .and.
     &          (jb .le. NTYBricks-ntybricksremove))   then
            joff = (jb-ntybricksremove - 1) * ny
            joff2 = 2 * joff
            do in = 1,ntxnodes
               MyTNode = (in - 1)  +  ntxnodes * (jn - 1
     &                             +  ntynodes * (kn - 1))
               mybb = MyTNode * NNBricks
               mybb = mybb  +  NNXBricks * (jnb - 1
     &                      +  NNYBricks * (knb - 1))
            do inb = 1,NNXBricks
               ib = (in - 1) * NNXBricks  +  inb
               ioff = (ib - 1) * nxby4
               ioff2 = 2 * ioff
               mybrick = mybb + inb
               ijk = 0
               do kk = 1,2*nz
                  k = koff2 + kk
                  do jj = 1,2*ny
                     j = joff2 + jj
                     do ii = 1,nxby2
                        i = ioff2 + ii
                        ijk = ijk + 1
                        iBoB8(i,j,k) = iBoB8brick(ijk,mybrick)
                     enddo
                  enddo
               enddo
            enddo
            enddo
            endif
         enddo
         enddo
      enddo
      enddo
c
      return
      end
#endif
c
c
c
c=======================================================================
c
c
c
c=======================================================================
c
c
c
      subroutine ReformatMoms (momsbricks,moms,
     &                         ncubesx,ncubesy,ncubesz,
     &                         ntybricksremove,
     &                         NTXBricks,NTYBricks,NTZBricks)
c
      integer*8   ncubesx,ncubesy,ncubesz,ntybricksremove
      integer*8   NTXBricks,NTYBricks,NTZBricks
c
      character*1  momsbricks((ncubesx*ncubesy*ncubesz/8)*48,
     &                        NTXBricks*NTYBricks*NTZBricks)
      character*1  moms((ncubesx*NTXBricks
     &                  *ncubesy*(NTYBricks-2*ntybricksremove)
     &                  *ncubesz*NTZBricks/8)*48)
c
      integer*8   nxby4,nxby2,nynxby4,ninbrick,mybrick,mybb
      integer*8   kn,knb,kb,koff,koff2,jn,jnb,jb,joff,joff2
      integer*8   in,inb,ib,ioff,ioff2
      integer*8   kk,k,jj,j,ii,i,ijk
      integer*8   NNXBricks,NNYBricks,NNZBricks,NNBricks
      integer*8   ntxnodes,ntynodes,ntznodes,MyTNode
c
      integer*8   len8,len8b,iprecision,iofflen,iofflenb
      integer*8   len,lenb,ncxh,ncyh,nczh,ivar
c
c     We have teams inside a domain, nodes in side a team, and bricks
c     inside a team.  This gives us the right assignments of MPI ranks
c     (bricks) to network nodes.  There is no advantage to doing this
c     careful assignment unless NTXBricks is even, and similarly for
c     Y and Z.  So we do it only in that case.
c
      NNXBricks = 2
      ntxnodes = NTXBricks / NNXBricks
      if (NNXBricks*ntxnodes .ne. NTXBricks)   NNXBricks = 1
      ntxnodes = NTXBricks / NNXBricks
      NNYBricks = 2
      ntynodes = NTYBricks / NNYBricks
      if (NNYBricks*ntynodes .ne. NTYBricks)   NNYBricks = 1
      ntynodes = NTYBricks / NNYBricks
      NNZBricks = 2
      ntznodes = NTZBricks / NNZBricks
      if (NNZBricks*ntznodes .ne. NTZBricks)   NNZBricks = 1
      ntznodes = NTZBricks / NNZBricks
      NNBricks = NNXBricks * NNYBricks * NNZBricks
c
c      MyTBrick = myrank - myteam*ntbricks
c
c      mytnode = MyTBrick / NNBricks
c      mytnodeZ = mytnode / (ntxnodes * ntynodes)
c      mytnodeX = mytnode  -  mytnodeZ * ntxnodes * ntynodes
c      mytnodeY = mytnodeX / ntxnodes 
c      mytnodeX = mytnodeX  -  mytnodeY * ntxnodes
c
c     The following coordinates are 1-based rather than 0-based.
c
c      MyTNBrick = MyTBrick  -  mytnode * NNBricks
c      MyTNBrickZ = MyTNBrick  / (NNXBricks * NNYBricks)
c      MyTNBrickX = MyTNBrick  -  MyTNBrickZ * NNXBricks * NNYBricks
c      MyTNBrickZ = MyTNBrickZ + 1
c      MyTNBrickY = MyTNBrickX / NNXBricks
c      MyTNBrickX = (MyTNBrickX + 1) -  MyTNBrickY * NNXBricks
c      MyTNBrickY = MyTNBrickY + 1
c
c      MyBrick = (MyTNBrickX-1) +  NNXBricks * (MyTNBrickY - 1
c     &                         +  NNYBricks * (MyTNBrickZ - 1))
c     &                         +  MyTNode * NNBricks
c
c     The above is a 0-based numbering for the bricks, like the MPI
c     ranks.  In the code below, we need a 1-based numbering.
c
      len8 = ncubesx*NTXBricks
     &      *ncubesy*(NTYBricks-2*ntybricksremove)
     &      *ncubesz*NTZBricks
      len8b = ncubesx*ncubesy*ncubesz
c
      do iprecision = 1,2
      iofflen = len8 * (iprecision - 1)
      iofflenb = len8b * (iprecision - 1)
      do kn = 1,ntznodes
      do knb = 1,NNZBricks
         kb = (kn - 1 ) * NNZBricks  +  knb
         koff = (kb - 1) * ncubesz
         do jn = 1,ntynodes
         do jnb = 1,NNYBricks
            jb = (jn - 1) * NNYBricks  +  jnb
            if ((jb .gt. ntybricksremove) .and.
     &          (jb .le. NTYBricks-ntybricksremove))   then
            joff = (jb-ntybricksremove - 1) * ncubesy
            do in = 1,ntxnodes
               MyTNode = (in - 1)  +  ntxnodes * (jn - 1
     &                             +  ntynodes * (kn - 1))
               mybb = MyTNode * NNBricks
               mybb = mybb  +  NNXBricks * (jnb - 1
     &                      +  NNYBricks * (knb - 1))
            do inb = 1,NNXBricks
               ib = (in - 1) * NNXBricks  +  inb
               ioff = (ib - 1) * ncubesx
               mybrick = mybb + inb
               ijk = iofflenb
               do k = 1,ncubesz
               do j = 1,ncubesy
                  iib = iofflen  +  ioff
     &                +  ncubesx * NTXBricks * (joff + j - 1
     &                +  ncubesy * (NTYBricks - 2*ntybricksremove)
     &                           * (koff + k - 1))
               do i = 1,ncubesx
                  ijk = ijk + 1
                  moms(iib+i) = momsbricks(ijk,mybrick)
               enddo
               enddo
               enddo
            enddo
            enddo
            endif
         enddo
         enddo
      enddo
      enddo
      enddo
c
      len = len8 / 8
      lenb = len8b / 8
      ncxh = ncubesx / 2
      ncyh = ncubesy / 2
      nczh = ncubesz / 2
c
      do ivar = 17,48
      iofflen = len * (ivar - 1)
      iofflenb = lenb * (ivar - 1)
      do kn = 1,ntznodes
      do knb = 1,NNZBricks
         kb = (kn - 1 ) * NNZBricks  +  knb
         koff = (kb - 1) * nczh
         do jn = 1,ntynodes
         do jnb = 1,NNYBricks
            jb = (jn - 1) * NNYBricks  +  jnb
            if ((jb .gt. ntybricksremove) .and.
     &          (jb .le. NTYBricks-ntybricksremove))   then
            joff = (jb-ntybricksremove - 1) * ncyh
            do in = 1,ntxnodes
               MyTNode = (in - 1)  +  ntxnodes * (jn - 1
     &                             +  ntynodes * (kn - 1))
               mybb = MyTNode * NNBricks
               mybb = mybb  +  NNXBricks * (jnb - 1
     &                      +  NNYBricks * (knb - 1))
            do inb = 1,NNXBricks
               ib = (in - 1) * NNXBricks  +  inb
               ioff = (ib - 1) * ncxh
               mybrick = mybb + inb
               ijk = iofflenb
               do k = 1,nczh
               do j = 1,ncyh
                  iib = iofflen  +  ioff
     &                +  ncxh * NTXBricks * (joff + j - 1
     &                +  ncyh * NTYBricks * (koff + k - 1))
               do i = 1,ncxh
                  ijk = ijk + 1
                  moms(iib+i) = momsbricks(ijk,mybrick)
               enddo
               enddo
               enddo
            enddo
            enddo
            endif
         enddo
         enddo
      enddo
      enddo
      enddo
c
      return
      end
