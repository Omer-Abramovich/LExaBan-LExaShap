SELECT 
	journal.full_name 
FROM 
	author, organization, writes, publication, journal, domain_journal, domain 
WHERE
	author.oid = organization.oid
	AND author.aid = writes.aid
    AND writes.pid = publication.pid
    AND publication.jid = journal.jid
    AND journal.jid = domain_journal.jid
    AND domain_journal.did = domain.did
    AND author.paper_count > 10
    AND organization.name = 'Tel Aviv University'
    AND publication.year > 2010
    AND publication.citation_count > 1
GROUP BY journal.full_name 