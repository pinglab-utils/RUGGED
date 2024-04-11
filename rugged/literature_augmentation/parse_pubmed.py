import re, itertools, json, sys, os, time, datetime, traceback
from lxml import etree

# Parses the PubMed documents
class Parser(object):
    def __init__(self, file, pubmed_output_file, filestat_output_file, parsing_config):
        # Input
        self.file = file
        self.fname = file.split('/')[-1].split('.')[0]

        self.pubmed_output_file = pubmed_output_file
        self.filestat_output_file = filestat_output_file
        self.parsing_config = parsing_config
        self.full_text_data = dict() #json.load(open("caseolap/pmid_fulltext_v1.json","r"))

        # Containers
        self.filestat = {}
        self.result = {}


    '''''''''''''''''
    Getter functions
    '''''''''''''''''
    # Get text
    def get_text(self, element, tag):
        '''
        FUNCTION:
        - Get the text from an xml element. This is used for
          different elements (e.g., author name, PMID)

        PARAMS:
        - element: The element/node in the xml structure
        - tag: The name of the element's section (e.g., 'LastName')
        '''
        e = element.find(tag)
        if e is not None:
            return e.text
        else:
            return ''


    # Get author names (0+ occurences)
    def parse_author(self, authors):
        '''
        FUNCTION:
        - Get the author names.

        PARAMS:
        - authors:

        OUTPUT:
        - author_list: A list of dictionaries of parts of an author's name
        '''
        author_list = []

        # Get the full name of each author
        for author in authors:
            author_name = dict()
            author_name['LastName'] = self.get_text(author, 'LastName')
            author_name['ForeName'] = self.get_text(author, 'ForeName')
            author_name['Initials'] = self.get_text(author, 'Initials')
            author_name['Suffix']   = self.get_text(author, 'Suffix')
            author_name['CollectiveName'] = self.get_text(author, 'CollectiveName')
            author_list.append(author_name)
        return author_list


    # Get the PubMed article ID (1 occurrence)
    def get_pmid(self, article):
        '''
        FUNCTION:
        - Get the article PubMed ID

        PARAMS:
        - article: element in the xml structure

        OUTPUT:
        - pmid: The PubMed ID
        '''
        pmid = self.get_text(article, './/PMID')
        self.result['PMID'] = pmid
        return pmid


    # Get the article title (0 or 1 occurrence)
    def get_title(self, article):
        '''
        FUNCTION:
        - Get the article title.
        - Save it to the class object

        PARAMS:
        - article:
        '''
        self.result['ArticleTitle'] = self.get_text(article, './/ArticleTitle')


    # Get the abstract (0 or 1 occurrence)
    def get_abstract(self, article, pmid):
        '''
        FUNCTION:
        - Get the article abstract
        - Save it to the class object

        PARAMS:
        - article: xml element
        - pmid (str): PubMed ID
        '''
        # Set default of abstract to be nothing
        self.result['Abstract'] = ''

        # Find if there is an abstract
        abstractList = article.find('.//Abstract')
        if abstractList != None:

            # Get the abstract
            try:
                abs_text = [line.text for line in abstractList.findall('AbstractText')]
                abstract = '\n'.join(abs_text)
                self.result['Abstract'] = abstract
            except:
                pass


    # Get the publishing date
    def get_publishing_date(self, journal):
        '''
        FUNCTION:
        - Get the date the article was published
        - Save it to the class object

        PARAMS:
        - journal:
        '''
        pre = './/JournalIssue/PubDate/'
        if self.parsing_config['date']:
            self.result['PubDate'] = dict()
            self.result['PubDate']['Year']   = self.get_text(journal, pre+'Year')
            self.result['PubDate']['Month']  = self.get_text(journal, pre+'Month')
            self.result['PubDate']['Day']    = self.get_text(journal, pre+'Day')
            self.result['PubDate']['Season'] = self.get_text(journal, pre+'Season')
            self.result['PubDate']['MedlineDate'] = self.get_text(journal,
                                                                  pre+'MedlineDate')


    # Get the MeSG Headings (0+ occurrences)
    def get_MeSH(self, article, pmid):
        '''
        FUNCTION:
        - Get the article MeSH term metadata (terms provided by PubMed
          in a separate section of the article. MeSH terms are topics the
          publication discusses)
          - Save it to the class object

        PARAMS:
        - article:
        - pmid (str): the PubMed ID of the article
        '''
        # Get MeSH heading elements
        headings = article.findall('.//MeshHeading')
        self.result['MeshHeadingList'] = []

        # If there are MeSH headings
        if headings:

            # Get all the MeSH headings
            for heading in headings:
                descriptor_names = heading.findall('DescriptorName')
                qualifier_names = heading.findall('QualifierName')

                # Get all MeSH descriptors
                if descriptor_names:
                    for descriptor_name in descriptor_names:
                        self.result['MeshHeadingList'].append(descriptor_name.text)

                # Get all MeSH qualifiers
                if qualifier_names:
                    for qualifier_name in qualifier_names:
                        self.result['MeshHeadingList'].append(qualifier_name.text)



    # Get statistics on the number of files
    def get_filestat(self, filepmids):
        '''
        FUNCTION:
        - Get count on the number of PubMed articles in a file
        - Save it to the class object

        PARAMS:
        - filepmids: List of PubMed IDs
        '''
        unique_filepmids = list(set(filepmids))
        file_counts = len(unique_filepmids)
        self.filestat.update({'fname':str(self.fname),
                              'pmids':file_counts})


    # Get the full article text (0 or 1 occurence)
    def get_fulltext(self):
        '''
        FUNCTION:
        - Get the article's full text if available (not just the abstract)
        - Save it to the class object
        '''
        # Set the default of the full text
        self.result["full_text"] = ''

        # If there is full text downloaded
        if self.result["PMID"] in self.full_text_data:
            full_text = self.full_text_data[self.result["PMID"]]
            if full_text is not None:
                self.result["full_text"] = full_text

    '''
    Parse one bulk file of PubMed publications
    '''
    ### Parse pubmed ###
    def parse_pubmed_file(self, logfile):
        '''
        FUNCTION:
        - This parses through articles in a bulk. It gets information from
          multiple fields (e.g., PMID, title, abstract, MeSH) and stores
          them in the class object.

        PARAMS:
        - logfile: Log file tracking the parsing progress
        '''
        sys.stdout.flush()
        t1 = time.time()

        # Parse the publications' .xml formats
        tree_file = open(self.file, 'r', encoding="utf8")
        tree = etree.parse(tree_file)

        filepmids = []
        full_text_data = self.full_text_data

        # Get all PubMed articles and documents
        articles = itertools.chain(tree.findall('PubmedArticle'),
                                   tree.findall('BookDocument'))

        # Get info on each article
        for article_count, article in enumerate(articles):

            # Get PMID
            pmid = self.get_pmid(article)
            filepmids.append(pmid)

            # Get Title
            if self.parsing_config['title']:
                self.get_title(article)

            # Get Abstract
            if self.parsing_config['abstract']:
                self.get_abstract(article,pmid)

            # Get MeSH
            if self.parsing_config['MeSH']:
                self.get_MeSH(article,pmid)

            # Get full text (if available)
            if self.parsing_config['full_text']:
                self.get_fulltext()

            # Author List
            if self.parsing_config['author']:
                authors = article.findall('.//Author')
                self.result['AuthorList'] = self.parse_author(authors)

            # Journal
            if self.parsing_config['journal']:
                journal = article.find('.//Journal')
                self.result['Journal'] = self.get_text(journal, 'Title')

            # Publishing Date
            if self.parsing_config['date']:
                self.get_publishing_date(journal)

            # Country Published
            if self.parsing_config['location']:
                country = article.find('.//MedlineJournalInfo')
                self.result['Country'] = self.get_text(country, 'Country')

            # Dump to the PubMed json file
            json.dump(self.result, self.pubmed_output_file)
            self.pubmed_output_file.write('\n')


        self.get_filestat(filepmids)

        # Dump to the PubMed stat json file
        json.dump(self.filestat, self.filestat_output_file)
        self.filestat_output_file.write('\n')
        t2 = time.time()

        msg = 'Parsing finished. '+'Total time: ' + str(t2 - t1)
        print(msg)
        logfile.write(msg + '\n')
        tree_file.close()


'''
Parse all bulks, main parsing function
'''
def parse_dir(source_dir, pubmed_output_file, filestat_output_file,
              ndir, parsing_config, logfile):
    '''
    FUNCTION:
    - This iterates through the directory containing the downloaded text. It then calls
      the above functions to parse the articles, taking the relevant fields (e.g.,
      title, abstract, etc.) and storing them in one dictionary for all articles.

    PARAMS:
    - source_dir (str): Where all the bulk PubMed articles were downloaded to
    - pubmed_output_file (text wrapper): Where the parsed PubMed data will be stored
    - parsing_config (dict): A dictionary indicating which fields to parse
    - filestat_output_file (text wrapper): Where article stats are stored
    - ndir (str): 'baseline' or 'updatefiles', name of directory
    - logfile (text wrapper): Log file of the parsing progress
    '''

    ''' Pre-parsing '''
    # Get current year so you can download the files from the latest year
    currentDateTime = datetime.datetime.now()
    date = currentDateTime.date()
    year = date.strftime("%Y")
    YEAR_LAST_TWO = year[-2:]

    # Finds total .xml files in the directory. The xml files will be parsed.
    total_files = 0
    for filename in os.listdir(source_dir):
        file = os.path.join(source_dir, filename)
        if os.path.isfile(file) and file.endswith('.xml'):
            total_files += 1

    # Print progress: starting
    msg = 'Parsing '+ndir+' containing '+str(total_files)+' bulk files\n'
    logfile.write(msg + '='*50 +'\n')
    print(msg +'='*50)


    ''' Parsing '''
    # Iterate through files
    if os.path.isdir(source_dir):
        file_count = 0

        # For each file in the baseline or update file directory
        for file in os.listdir(source_dir):

            # Parse if it is is an .xml file
            # NOTE: Check that the year and file name is correct here.
            # Sometimes they change both. Check the ftp baseline server
            if re.search(r'^pubmed'+YEAR_LAST_TWO+'n\d\d\d\d.xml$', file) is not None:
                
                # Parse
                PRS = Parser(os.path.join(source_dir, file), pubmed_output_file,
                             filestat_output_file, parsing_config)
                PRS.parse_pubmed_file(logfile)

                # Print progress
                file_count += 1
                msg='Parsing '+str(file_count)+'/'+str(total_files)+' from '\
                    + ndir + ':' + file
                logfile.write(msg + "\n")
                print(msg)