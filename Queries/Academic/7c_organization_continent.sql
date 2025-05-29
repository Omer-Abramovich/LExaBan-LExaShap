SELECT 
	organization.continent 
FROM 
	organization, author as author1, writes as writes1, author as author2, writes as writes2, publication, journal, domain_journal, domain 
WHERE
	organization.oid = author1.oid 
	AND organization.oid = author2.oid
    AND author1.name < author2.name
    AND author1.aid = writes1.aid
    AND author2.aid = writes2.aid
    AND writes1.pid = publication.pid
    AND writes2.pid = publication.pid
    AND publication.jid = journal.jid
    AND journal.jid = domain_journal.jid
    AND domain_journal.did = domain.did
    AND organization.citation_count > 5000
    AND publication.citation_count > 10    
    AND domain.name = 'Databases'
    and publication.year > 2010
GROUP BY organization.continent 